"""Sprint 4 backend: a small FastAPI server wrapping the analysis pipeline.

Three ways to get results:
  - GET  /games              list the rulebooks already analysed (instant, no API cost)
  - GET  /games/{id}         that game's saved analysis + FAQ (instant)
  - GET  /pdf/{id}           the original source PDF, so a finding's page number can
                              link straight to the real page (NFR3, Chapter 3.6)
  - POST /analyze            upload a new PDF, run the full pipeline live (costs a
                              few cents and takes ~30-90s, since it calls the Claude API);
                              the optional form field "model" is "haiku" (default) or "sonnet",
                              "doc_language" the language of the rulebook (default: detected) and
                              "report_language" the language of the explanations (default: English)
  - POST /detect-language    the language of an uploaded PDF, found from its first pages, so that the
                              interface can show it before the analysis (no API call, nothing is stored)
  - POST /ask                a question about an analysed rulebook, in any language, answered from the
                              rulebook only, with its pages (costs a cent or two)

Run with:
    uvicorn api:app --reload --port 8000
"""
import json
import os
import shutil
import tempfile
import urllib.parse
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import analyze_rulebook
import detect
import generate_faq
import language as language_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
ANALYSIS_DIR = os.path.join(BASE_DIR, "output", "analysis")
UPLOAD_DIR = os.path.join(BASE_DIR, "output", "uploads")

app = FastAPI(title="Board Game Rulebook Validator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-only convenience; a deployed version should restrict this
    allow_methods=["*"],
    allow_headers=["*"],
)


def game_id(source_filename):
    """A short, URL-safe id derived from a rulebook's stored source filename."""
    return os.path.splitext(source_filename)[0]


def inline_content_disposition(filename):
    """A Content-Disposition value safe for any filename, including non-ASCII ones
    (e.g. the en-dash in "CATAN-The Game Rulebook secure.pdf").

    HTTP header values must be latin-1 (RFC 7230), so a raw Unicode filename in the
    header crashes the response with a 500 (found via testing every dataset game
    through this endpoint, not just the ASCII-named ones - see Chapter 5). The
    RFC 5987 filename* parameter carries the real UTF-8 name; the plain filename
    stays as an ASCII-only fallback for older clients that ignore filename*.
    """
    ascii_fallback = filename.encode("ascii", "ignore").decode("ascii").strip() or "rulebook.pdf"
    encoded = urllib.parse.quote(filename)
    return f'inline; filename="{ascii_fallback}"; filename*=UTF-8\'\'{encoded}'


def load_result(stem):
    analysis_path = os.path.join(ANALYSIS_DIR, f"{stem}.analysis.json")
    faq_path = os.path.join(ANALYSIS_DIR, f"{stem}.faq.json")
    if not os.path.exists(analysis_path):
        return None
    with open(analysis_path, encoding="utf-8") as f:
        result = json.load(f)
    if os.path.exists(faq_path):
        with open(faq_path, encoding="utf-8") as f:
            faq_data = json.load(f)
        result["faq"] = faq_data.get("faq", [])
        result["needs_human_review"] = faq_data.get("needs_human_review", [])
    else:
        result["faq"] = []
        result["needs_human_review"] = [
            i for i in result["issues"] if i["category"] == "contradiction"
        ]
    return result


@app.get("/games")
def list_games():
    """Every rulebook that already has a saved analysis, for instant loading in the demo."""
    games = []
    for fname in sorted(os.listdir(ANALYSIS_DIR)):
        if fname.endswith(".analysis.json"):
            stem = fname.replace(".analysis.json", "")
            with open(os.path.join(ANALYSIS_DIR, fname), encoding="utf-8") as f:
                d = json.load(f)
            games.append({
                "id": stem,
                "source": d["source"],
                "issue_count": len(d["issues"]),
                "pages": d["stats"]["pages"],
            })
    return games


@app.get("/games/{game_id}")
def get_game(game_id: str):
    result = load_result(game_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No saved analysis for '{game_id}'")
    return result


@app.get("/pdf/{game_id}")
def get_pdf(game_id: str):
    """Serve the original source PDF for a game, so a finding's page reference can
    open the real page. Checks the built-in dataset first, then live-uploaded PDFs -
    a game's "source" filename lives in exactly one of the two (see analyze_rulebook.
    parse_and_tag and api.analyze_pdf)."""
    result = load_result(game_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No saved analysis for '{game_id}'")
    source = result["source"]
    for directory in (DATASET_DIR, UPLOAD_DIR):
        candidate = os.path.join(directory, source)
        if os.path.exists(candidate):
            # FileResponse defaults to Content-Disposition: attachment when a filename
            # is given, which makes the browser download the file instead of opening
            # it - that would silently break the #page=N jump the frontend relies on.
            # "inline" tells the browser to open it in its own PDF viewer instead.
            return FileResponse(
                candidate,
                media_type="application/pdf",
                headers={"Content-Disposition": inline_content_disposition(source)},
            )
    raise HTTPException(status_code=404, detail=f"Source PDF file for '{game_id}' was not found on disk")


@app.post("/detect-language")
async def detect_language_of_pdf(file: UploadFile = File(...)):
    """The language of an uploaded PDF, from the text of its first pages. Free and quick; the file is not kept.

    confidence is "high", "low" (the interface shows the language as uncertain) or "none" (language is null:
    a scanned file or too little text, and the user has to choose)."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp.flush()
        try:
            found = language_support.detect_pdf_language(tmp.name)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"The PDF could not be read: {e}")
    return {"language": found["language"], "confidence": found["confidence"], "languages": language_support.LANGUAGES}


@app.post("/analyze")
async def analyze_pdf(file: UploadFile = File(...), model: str = Form(analyze_rulebook.FINAL_MODEL),
                      doc_language: str = Form("auto"), report_language: str = Form(language_support.DEFAULT_LANGUAGE)):
    """Run the full pipeline on a freshly uploaded PDF. Costs a few cents and ~30-90s.

    model: "haiku" (the default, the model of the evaluation) or "sonnet". Any other value is
    refused before the file is stored, so a client can never make the server call a model that
    is not in detect.MODELS.
    doc_language / report_language: one of language.LANGUAGES (a name or a code). "auto" for doc_language detects the
    language of the rulebook. The rulebook is analysed in its own language; the explanations and the FAQ are written in
    the report language. A value that is not a supported language is refused in the same way as a model.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")
    if model not in detect.MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model. Choose one of: {', '.join(detect.MODELS)}")
    try:
        report_language = language_support.normalize_language(report_language)
        doc_language = language_support.normalize_language(doc_language, default=None)  # None: detect it from the PDF
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
    dest_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(dest_path, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        analysis = analyze_rulebook.analyze(dest_path, model=model, doc_language=doc_language, report_language=report_language)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    stem = os.path.splitext(analysis["source"])[0]

    import anthropic
    client = anthropic.Anthropic()
    faq, review_queue, _usage = generate_faq.generate_faq_entries(
        client, analysis["issues"], quiet=True, model=model,
        doc_language=analysis["language"]["document"], report_language=report_language)

    with open(os.path.join(ANALYSIS_DIR, f"{stem}.faq.json"), "w", encoding="utf-8") as f:
        json.dump({"source": analysis["source"], "model": analysis["model"], "language": analysis["language"], "faq": faq,
                   "needs_human_review": review_queue}, f, ensure_ascii=False, indent=2)

    analysis["faq"] = faq
    analysis["needs_human_review"] = review_queue
    return analysis


class AskRequest(BaseModel):
    game_id: str
    question: str
    model: str = analyze_rulebook.FINAL_MODEL


MAX_QUESTION_CHARS = 500


def load_sentences(stem, result):
    """The sentences of an analysed rulebook, as they were extracted: from the file kept next to the analysis, from
    the output of the Sprint 2 pipeline (the dataset rulebooks), or, for an older analysis, by reading the PDF again
    (the result is then kept for the next question)."""
    kept = os.path.join(ANALYSIS_DIR, f"{stem}.sentences.json")
    if os.path.exists(kept):
        with open(kept, encoding="utf-8") as f:
            return json.load(f)["sentences"]
    sprint2 = os.path.join(BASE_DIR, "output", f"{os.path.splitext(result['source'])[0]}.nlp.json")
    if os.path.exists(sprint2):
        with open(sprint2, encoding="utf-8") as f:
            return json.load(f)["sentences"]
    for directory in (DATASET_DIR, UPLOAD_DIR):
        pdf_path = os.path.join(directory, result["source"])
        if os.path.exists(pdf_path):
            language = (result.get("language") or {}).get("document", language_support.DEFAULT_LANGUAGE)
            sentences = analyze_rulebook.parse_and_tag(pdf_path, language)["sentences"]
            with open(kept, "w", encoding="utf-8") as f:
                json.dump({"source": result["source"], "language": language, "sentences": sentences}, f, ensure_ascii=False)
            return sentences
    raise HTTPException(status_code=404, detail=f"The text of '{stem}' is not available")


@app.post("/ask")
def ask_about_rulebook(req: AskRequest):
    """Answer a player's question about an analysed rulebook from the rulebook only.

    The question can be in any language and is answered in its language. The answer cites sentences of the rulebook; the
    pages and the quotes come from the rulebook. When the rulebook does not give enough information the answer says so
    ("answerable": false) instead of guessing. Costs a cent or two (the whole text is sent).
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please write a question")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(status_code=400, detail=f"The question is too long (at most {MAX_QUESTION_CHARS} characters)")
    if req.model not in detect.MODELS:
        raise HTTPException(status_code=400, detail=f"Unknown model. Choose one of: {', '.join(detect.MODELS)}")
    result = load_result(req.game_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No saved analysis for '{req.game_id}'")
    sentences = load_sentences(req.game_id, result)

    import anthropic
    try:
        answer = generate_faq.answer_question(anthropic.Anthropic(), sentences, result["source"], question, model=req.model)
    except anthropic.APIError as e:
        raise HTTPException(status_code=502, detail=f"The question could not be answered: {e}")
    answer["model"] = detect.MODELS[req.model]
    return answer


@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/games", "/games/{id}", "/pdf/{id}", "POST /analyze", "POST /detect-language", "POST /ask"]}
