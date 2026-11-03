"""Checks of the multilingual support (1 November 2026): language detection, the language note of the prompts, the NLP
stage for other languages, the endpoints and the answers to questions.

Like scenario_checks.py, none of the checks calls the paid API: the client is a stub, and the HTTP operations are called
through FastAPI's test client with the folders redirected to a temporary directory. The test rulebooks are the two-page
PDFs of an invented card game in fixtures/ (the same planted problems in each of the nine languages).

Usage (from the src folder):
    python3 language_checks.py
"""
import contextlib
import glob
import io
import json
import os
import re
import sys
import tempfile
import time
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import anthropic
import analyze_rulebook
import detect
import detect_contradictions
import generate_faq
import language as L
import api
from fastapi.testclient import TestClient

FIXTURES = os.path.join(BASE_DIR, "fixtures")
SAMPLES = json.load(open(os.path.join(FIXTURES, "language_samples.json"), encoding="utf-8"))
pdf_of = lambda language: os.path.join(FIXTURES, f"Kartenjagd_{language}.pdf")
text_of = lambda language, sections=None: " ".join(h + ". " + t for h, t in (SAMPLES[language] if sections is None else SAMPLES[language][:sections]))
results = []


def check(cid, scenario, expected, fn):
    t0 = time.time()
    try:
        ok, actual = fn()
    except Exception as e:
        ok, actual = False, f"exception: {type(e).__name__}: {str(e)[:120]}"
    results.append(dict(id=cid, scenario=scenario, expected=expected, actual=actual, passed=bool(ok), seconds=round(time.time() - t0, 2)))
    print(f"{'PASS' if ok else 'FAIL'}  {cid:4} {scenario[:72]:72} -> {actual[:80]}")


# ------------------------------------------------------------------ detection of the language
def l1():
    got = {lang: L.detect_language(text_of(lang)) for lang in L.LANGUAGES}
    bad = {lang: r["language"] for lang, r in got.items() if r["language"] != lang or r["confidence"] != "high"}
    return not bad, "9 of 9 correct, all with high confidence" if not bad else f"wrong: {bad}"


def l2():
    bad = {lang: L.detect_language(text_of(lang, 1))["language"] for lang in L.LANGUAGES if L.detect_language(text_of(lang, 1))["language"] != lang}
    return not bad, "9 of 9 correct from one section of about 40 words" if not bad else f"wrong: {bad}"


def l3():
    bad = {}
    for lang in L.LANGUAGES:
        r = L.detect_pdf_language(pdf_of(lang))
        if r["language"] != lang:
            bad[lang] = r["language"]
    return not bad, "9 of 9 correct from the PDF files" if not bad else f"wrong: {bad}"


def l4():
    pdfs = sorted(glob.glob(os.path.join(BASE_DIR, "datasets", "*.pdf")))
    if not pdfs:
        return True, "no dataset PDFs here, skipped"
    got = {os.path.basename(p)[:12]: L.detect_pdf_language(p)["language"] for p in pdfs}
    return set(got.values()) == {"English"}, f"{len(pdfs)} real rulebooks: {set(got.values())}"


def l5():
    r1, r2 = L.detect_language("Hello"), L.detect_language("12 34 56 78 90 A B C D")
    return r1["language"] is None and r2["language"] is None, f"short text and numbers: {r1['confidence']}, {r2['confidence']}"


def l6():
    ok = (L.normalize_language("german") == "German" and L.normalize_language("SQ") == "Albanian" and L.normalize_language("") == "English"
          and L.normalize_language("auto", default=None) is None and L.normalize_language(None) == "English")
    try:
        L.normalize_language("Klingon")
        return False, "an unknown language was accepted"
    except ValueError:
        return ok, "names and codes accepted, empty and auto give the default, an unknown language raises"


# ------------------------------------------------------------------ the prompts
def stub_client(tool_input_for):
    """A client that records every call; tool_input_for(call kwargs) gives the tool input of the answer."""
    calls = []

    def create(**kw):
        calls.append(kw)
        block = SimpleNamespace(type="tool_use", input=tool_input_for(kw))
        return SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


def p1():
    notes = [L.language_note("English", "English"), L.language_note(None, None), L.language_note("English", "English", "faq")]
    de_en, sq_sq = L.language_note("German", "English"), L.language_note("Albanian", "Albanian")
    ok = (notes == ["", "", ""] and "German" in de_en and "English" in de_en and "Albanian" in sq_sq
          and "suggested_rewrite\" in German" in de_en and "explanation\" in English" in de_en)
    return ok, "empty for English with an English report; the other notes name both languages"


def p2():
    client, calls = stub_client(lambda kw: {"issues": []})
    detect.call_claude(client, "m", "text")
    detect.call_claude(client, "m", "text", L.language_note("English", "English"))
    detect.call_claude(client, "m", "text", L.language_note("German", "Albanian"))
    same = calls[0]["system"] == detect.SYSTEM_PROMPT and calls[1]["system"] == detect.SYSTEM_PROMPT
    note = calls[2]["system"]
    ok = same and note.startswith(detect.SYSTEM_PROMPT + "\n\n") and "German" in note and "Albanian" in note
    return ok, "no note: the original prompt, byte for byte; a note is added after it"


def p3():
    client, calls = stub_client(lambda kw: {"contradictions": []})
    detect_contradictions.call_claude(client, "m", "text")
    detect_contradictions.call_claude(client, "m", "text", L.language_note("Greek", "English"))
    ok = calls[0]["system"] == detect_contradictions.SYSTEM_PROMPT and calls[1]["system"].startswith(detect_contradictions.SYSTEM_PROMPT + "\n\n") \
        and "Greek" in calls[1]["system"]
    return ok, "whole-document pass: the original prompt, or the prompt followed by the note"


def p4():
    issue = dict(id=0, category="unclear_wording", pages=[1], quote="q", explanation="e", suggested_rewrite="r", severity="low")
    client, calls = stub_client(lambda kw: {"entries": [{"issue_number": 1, "question": "Q?", "answer": "A.", "confidence": "high"}]})
    generate_faq.generate_faq_entries(client, [issue], quiet=True)
    generate_faq.generate_faq_entries(client, [issue], quiet=True, doc_language="English", report_language="English")
    generate_faq.generate_faq_entries(client, [issue], quiet=True, doc_language="German", report_language="Albanian")
    ok = calls[0]["system"] == generate_faq.SYSTEM_PROMPT == calls[1]["system"] and calls[2]["system"].startswith(generate_faq.SYSTEM_PROMPT + "\n\n") \
        and "Albanian" in calls[2]["system"] and "German" in calls[2]["system"]
    return ok, "FAQ stage: the original prompt, or the prompt followed by the note"


# ------------------------------------------------------------------ the NLP stage for other languages
def n1():
    bad = {}
    for lang in L.LANGUAGES:
        tagged = analyze_rulebook.parse_and_tag(pdf_of(lang), lang)
        clean = [s for s in tagged["sentences"] if not s["noise"]]
        if tagged["stats"]["noise_sentences"] or {s["page"] for s in clean} != {1, 2} or len(clean) < 10:
            bad[lang] = (tagged["stats"]["noise_sentences"], len(clean))
    return not bad, "no sentence is dropped as noise, pages 1 and 2 kept (9 languages)" if not bad else f"problems: {bad}"


def n2():
    pdf = os.path.join(BASE_DIR, "datasets", "Carcassonne_20th_Anniversary_Rulebook.pdf")
    reference = os.path.join(BASE_DIR, "output", "Carcassonne_20th_Anniversary_Rulebook.nlp.json")
    if not (os.path.exists(pdf) and os.path.exists(reference)):
        return True, "no dataset files here, skipped"
    new, ref = analyze_rulebook.parse_and_tag(pdf), json.load(open(reference, encoding="utf-8"))
    same = [(s["text"], s["noise"], s["page"]) for s in new["sentences"]] == [(s["text"], s["noise"], s["page"]) for s in ref["sentences"]]
    return same and new["issues"] == ref["issues"], f"English: {len(new['sentences'])} sentences, the same text, noise and pages as the saved Sprint 2 output"


# ------------------------------------------------------------------ the analysis with a stub client
def echo_first_sentence(kw):
    """The answer of a stub model: one finding that quotes the first sentence of the text that it received."""
    if kw["tool_choice"]["name"] == "write_faq":
        return {"entries": [{"issue_number": 1, "question": "Question?", "answer": "Answer.", "confidence": "high"}]}
    first = re.search(r"\[S(\d+)\] (.+)", kw["messages"][0]["content"])
    sid, quote = int(first.group(1)), first.group(2).strip()
    item = dict(sentence_ids=[sid], quote=quote, explanation="explanation", severity="low", suggested_rewrite="rewrite")
    if kw["tool_choice"]["name"] == "report_issues":
        return {"issues": [dict(item, category="unclear_wording", is_extraction_artifact=False)]}
    return {"contradictions": []}


def analyse_with_stub(language, doc_language, report_language):
    client, calls = stub_client(echo_first_sentence)
    real_client = anthropic.Anthropic
    anthropic.Anthropic = lambda *a, **k: client
    folder = tempfile.mkdtemp()
    old = analyze_rulebook.RESULT_DIR
    analyze_rulebook.RESULT_DIR = folder
    try:
        with contextlib.redirect_stdout(io.StringIO()):                    # analyze() reports its progress; not needed here
            result = analyze_rulebook.analyze(pdf_of(language), doc_language=doc_language, report_language=report_language)
    finally:
        anthropic.Anthropic, analyze_rulebook.RESULT_DIR = real_client, old
    return result, calls, folder


def a1():
    bad = {}
    for lang in L.LANGUAGES:
        result, calls, folder = analyse_with_stub(lang, None, "Albanian" if lang != "Albanian" else "German")   # the language is detected
        report = "Albanian" if lang != "Albanian" else "German"
        systems = " ".join(c["system"] for c in calls)
        issue = result["issues"][0]
        sentences = json.load(open(os.path.join(folder, f"{os.path.splitext(result['source'])[0]}.sentences.json"), encoding="utf-8"))["sentences"]
        ok = (result["language"] == {"document": lang, "report": report} and (lang == "English" or lang in systems) and report in systems
              and issue["quote_verified"] is True and issue["pages"] == [1] and issue["quote"] == sentences[issue["sentence_ids"][0]]["text"])
        if not ok:
            bad[lang] = result["language"]
    return not bad, "9 of 9: language detected, the note names both, the quote is verified word for word" if not bad else f"problems: {bad}"


def a2():
    result, calls, _ = analyse_with_stub("English", None, "English")
    same = all(c["system"] in (detect.SYSTEM_PROMPT, detect_contradictions.SYSTEM_PROMPT) for c in calls)
    return same and result["language"] == {"document": "English", "report": "English"}, f"English rulebook, English report: {len(calls)} calls, all with the original prompts"


def a3():
    result, calls, _ = analyse_with_stub("German", "German", "English")   # a language that is given is used as it is
    return result["language"]["document"] == "German" and all("German" in c["system"] for c in calls), "a given language is used without detection"


# ------------------------------------------------------------------ the HTTP operations
uploads, results_dir = tempfile.mkdtemp(), tempfile.mkdtemp()
api.UPLOAD_DIR, api.ANALYSIS_DIR, analyze_rulebook.RESULT_DIR = uploads, results_dir, results_dir
client = TestClient(api.app)


def h1():
    bad = {}
    for lang in L.LANGUAGES:
        with open(pdf_of(lang), "rb") as f:
            r = client.post("/detect-language", files={"file": ("rulebook.pdf", f, "application/pdf")})
        if r.status_code != 200 or r.json()["language"] != lang or r.json()["languages"] != L.LANGUAGES:
            bad[lang] = r.json()
    return not bad and not os.listdir(uploads), "9 of 9 correct, the file is not kept" if not bad else f"problems: {bad}"


def h2():
    r1 = client.post("/detect-language", files={"file": ("notes.txt", b"hello", "text/plain")})
    r2 = client.post("/detect-language", files={"file": ("broken.pdf", b"this is not a pdf", "application/pdf")})
    return r1.status_code == 400 and r2.status_code == 400, f"HTTP {r1.status_code} for a text file, {r2.status_code} for a damaged PDF"


def h3():
    pdf = ("a.pdf", b"x", "application/pdf")
    r1 = client.post("/analyze", files={"file": pdf}, data={"report_language": "Klingon"})
    r2 = client.post("/analyze", files={"file": pdf}, data={"doc_language": "xx"})
    return r1.status_code == 400 and r2.status_code == 400 and not os.listdir(uploads), \
        f"HTTP {r1.status_code} and {r2.status_code}: an unknown language is refused before the file is stored"


def h4():
    seen = []
    real_analyze, real_faq = analyze_rulebook.analyze, generate_faq.generate_faq_entries

    def fake_analyze(path, model="haiku", doc_language=None, report_language="English"):
        seen.append((doc_language, report_language))
        return {"source": os.path.basename(path), "model": "m", "language": {"document": doc_language or "English", "report": report_language},
                "issues": [], "stats": {"pages": 1}}
    analyze_rulebook.analyze = fake_analyze
    generate_faq.generate_faq_entries = lambda *a, **k: ([], [], None)
    real_client = anthropic.Anthropic
    anthropic.Anthropic = lambda *a, **k: object()
    try:
        pdf = ("a.pdf", b"x", "application/pdf")
        client.post("/analyze", files={"file": pdf})                                                    # nothing given: detect, English
        client.post("/analyze", files={"file": pdf}, data={"doc_language": "de", "report_language": "sq"})
        client.post("/analyze", files={"file": pdf}, data={"doc_language": "auto", "report_language": "french"})
    finally:
        analyze_rulebook.analyze, generate_faq.generate_faq_entries, anthropic.Anthropic = real_analyze, real_faq, real_client
    return seen == [(None, "English"), ("German", "Albanian"), (None, "French")], f"{seen}"


# ------------------------------------------------------------------ questions of a player
SENTENCES = [
    {"id": 0, "page": 1, "section": "SETUP", "text": "Each player starts with 5 cards.", "noise": False},
    {"id": 1, "page": 2, "section": "TURN", "text": "Draw one card.", "noise": False},
    {"id": 2, "page": 2, "section": "TURN", "text": "tottaotlal inwinitwhith", "noise": True},
]


def ask_with(answer):
    client_, calls = stub_client(lambda kw: answer)
    return generate_faq.answer_question(client_, SENTENCES, "rules.pdf", "How many cards?"), calls


def q1():
    r, calls = ask_with({"answerable": True, "answer": "5 cards.", "sentence_ids": [0, 1, 99]})
    ok = (r["answerable"] and r["sentence_ids"] == [0, 1] and r["pages"] == [1, 2] and r["quotes"] == [SENTENCES[0]["text"], SENTENCES[1]["text"]]
          and "[S0] Each player starts with 5 cards." in calls[0]["messages"][0]["content"] and "[S2]" not in calls[0]["messages"][0]["content"])
    return ok, "pages and quotes come from the rulebook; an unknown id is dropped; noise is not sent"


def q2():
    r1, _ = ask_with({"answerable": True, "answer": "Six cards.", "sentence_ids": []})
    r2, _ = ask_with({"answerable": True, "answer": "Six cards.", "sentence_ids": [2, 99]})
    ok = all(not r["answerable"] and r["answer"] == generate_faq.NO_SUPPORT and r["pages"] == [] for r in (r1, r2))
    return ok, "an answer without a valid citation (none, unknown, noise) is never shown"


def q3():
    r, _ = ask_with({"answerable": False, "answer": "Das Regelheft sagt dazu nichts.", "sentence_ids": []})
    r2, _ = ask_with({"answerable": False, "answer": "", "sentence_ids": []})
    return (not r["answerable"] and r["answer"] == "Das Regelheft sagt dazu nichts." and r2["answer"] == generate_faq.NO_SUPPORT), \
        "a refusal keeps the words of the model (in the language of the question); an empty one gets the standard text"


def q4():
    json.dump({"source": "t.pdf", "language": "German", "sentences": SENTENCES}, open(os.path.join(results_dir, "t.sentences.json"), "w"))
    json.dump({"source": "t.pdf", "model": "m", "issues": [], "stats": {"pages": 2}}, open(os.path.join(results_dir, "t.analysis.json"), "w"))
    real_client = anthropic.Anthropic
    anthropic.Anthropic = lambda *a, **k: stub_client(lambda kw: {"answerable": True, "answer": "5.", "sentence_ids": [0]})[0]
    try:
        ok_r = client.post("/ask", json={"game_id": "t", "question": "Wie viele Karten bekommt jeder Spieler?"})
        empty = client.post("/ask", json={"game_id": "t", "question": "   "})
        long = client.post("/ask", json={"game_id": "t", "question": "x" * 501})
        model = client.post("/ask", json={"game_id": "t", "question": "q", "model": "gpt-4"})
        unknown = client.post("/ask", json={"game_id": "nope", "question": "q"})
    finally:
        anthropic.Anthropic = real_client
    d = ok_r.json()
    ok = (ok_r.status_code == 200 and d["answerable"] and d["pages"] == [1] and "model" in d
          and (empty.status_code, long.status_code, model.status_code, unknown.status_code) == (400, 400, 400, 404))
    return ok, f"HTTP 200 with the pages; 400 for an empty, a long question and a wrong model; {unknown.status_code} for an unknown game"


def q5():
    real_client = anthropic.Anthropic
    anthropic.Anthropic = lambda *a, **k: stub_client(lambda kw: {"answerable": True, "answer": "Yes.", "sentence_ids": [10]})[0]
    old = api.ANALYSIS_DIR
    try:
        api.ANALYSIS_DIR = os.path.join(BASE_DIR, "output", "analysis")        # an analysed dataset rulebook (no file next to it)
        if not os.path.exists(os.path.join(api.ANALYSIS_DIR, "Carcassonne_20th_Anniversary_Rulebook.analysis.json")):
            return True, "no saved Carcassonne analysis here, skipped"
        r = client.post("/ask", json={"game_id": "Carcassonne_20th_Anniversary_Rulebook", "question": "How many points does a monastery score?"})
    finally:
        anthropic.Anthropic, api.ANALYSIS_DIR = real_client, old
    return r.status_code == 200 and r.json()["pages"] and r.json()["sentence_ids"] == [10], f"HTTP {r.status_code}; the sentences of the dataset rulebook were found"


def q6():
    r, calls = ask_with({"answerable": True, "sentence_ids": [0, 1],
                         "answer": "Das Spiel beginnt im Uhrzeigersinn [S0]. Jeder bekommt 5 Karten (S0, S1) und zieht eine [S1–S2]."})
    only_numbers, _ = ask_with({"answerable": True, "answer": "[S0]", "sentence_ids": [0]})
    ok = (r["answer"] == "Das Spiel beginnt im Uhrzeigersinn. Jeder bekommt 5 Karten und zieht eine." and r["pages"] == [1, 2]
          and only_numbers["answer"] == generate_faq.NO_SUPPORT and not only_numbers["answerable"]
          and "cannot see the sentence numbers" in calls[0]["system"])
    return ok, "sentence numbers written into an answer are removed (the pages still come from sentence_ids); the prompt forbids them"


for cid, scenario, expected, fn in [
    ("L1", "Detect the language of the full text of each of the nine test rulebooks", "9 of 9, high confidence", l1),
    ("L2", "Detect the language from one section (about 40 words)", "9 of 9", l2),
    ("L3", "Detect the language from the PDF files", "9 of 9", l3),
    ("L4", "Detect the language of the five real rulebooks (English)", "English, all five", l4),
    ("L5", "A text that is too short or only numbers", "no language", l5),
    ("L6", "Languages given as names, codes, auto or unknown text", "normalised or refused", l6),
    ("P1", "The language note is empty for English with an English report", "empty; otherwise it names both languages", p1),
    ("P2", "System prompt of the chunk pass with and without a note", "the original, or the original + the note", p2),
    ("P3", "System prompt of the whole-document pass with and without a note", "the original, or the original + the note", p3),
    ("P4", "System prompt of the FAQ stage with and without a note", "the original, or the original + the note", p4),
    ("N1", "NLP stage on the nine test rulebooks with their own language", "no sentence dropped as noise", n1),
    ("N2", "NLP stage on an English dataset rulebook", "identical to the saved output of Sprint 2", n2),
    ("A1", "Analysis (stub client) of each test rulebook, language detected, report in Albanian or German", "language in the note and in the result", a1),
    ("A2", "Analysis of the English test rulebook with an English report", "the original prompts", a2),
    ("A3", "Analysis with a language that is given", "used without detection", a3),
    ("H1", "POST /detect-language with each test PDF", "9 of 9; nothing is stored", h1),
    ("H2", "POST /detect-language with a text file and a damaged PDF", "400", h2),
    ("H3", "POST /analyze with an unknown report or rulebook language", "400, nothing stored", h3),
    ("H4", "POST /analyze with language names and codes", "normalised names reach the pipeline", h4),
    ("Q1", "answer_question with a valid citation", "pages and quotes from the rulebook", q1),
    ("Q2", "answer_question that claims an answer without a valid citation", "not answerable", q2),
    ("Q3", "answer_question that says the rulebook does not answer", "refusal kept", q3),
    ("Q4", "POST /ask: a valid question and the invalid ones", "200 or 400/404", q4),
    ("Q5", "POST /ask for an analysed dataset rulebook", "the sentences are found", q5),
    ("Q6", "answer_question whose answer contains sentence numbers like [S42]", "the numbers are removed", q6),
]:
    check(cid, scenario, expected, fn)

if __name__ == "__main__":
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed} of {len(results)} checks passed")
    sys.exit(0 if passed == len(results) else 1)
