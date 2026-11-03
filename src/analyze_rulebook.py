"""End-to-end rulebook analysis pipeline (Sprint 3 deliverable).

Ties together every stage built so far into one command a future frontend can
call on a single PDF:

  parser.extract_pages          -> cleaned per-page text (Sprint 2)
  nlp_pipeline.analyse          -> sentences, structure, rule-based candidates (Sprint 2)
  detect.build_chunks/detect    -> LLM detection of unclear wording, terminology,
                                    timing and local contradictions (Sprint 3)
  detect_contradictions         -> whole-document contradiction pass, catches
                                    conflicts between distant pages (Sprint 3)

The default model is the result of the Sprint 3 evaluation (see
output/labeling/*.xlsx and the *.contradictions.json human_verdict fields):
Haiku, prompt v2, hints off - it had equal or better precision than Sonnet in
every rulebook we checked, at a fraction of the cost. analyze() also accepts
model="sonnet", which the web interface offers as an option on the upload page;
the evaluation of Chapter 6 was done with Haiku (and the saved analyses are Haiku).

Usage:
    python analyze_rulebook.py --pdf datasets/Carcassonne_20th_Anniversary_Rulebook.pdf
    python analyze_rulebook.py --game Carcassonne   # reuse an already-parsed dataset entry
"""
import argparse
import json
import os
import sys
import time

import anthropic
import spacy

import parser as rulebook_parser
import nlp_pipeline
import detect
import detect_contradictions
import language as language_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULT_DIR = os.path.join(OUTPUT_DIR, "analysis")

FINAL_MODEL = "haiku"  # decision from the Sprint 3 evaluation - see module docstring


def resolve_document_language(pdf_path, doc_language=None):
    """The language of the rulebook: the one that was given (a name or a code), or the one detected from the PDF.
    When nothing can be detected (a scanned file, very little text) the original default, English, is used."""
    if doc_language not in (None, "", "auto"):
        return language_support.normalize_language(doc_language)
    detected = language_support.detect_pdf_language(pdf_path)
    return detected["language"] or language_support.DEFAULT_LANGUAGE


def parse_and_tag(pdf_path, doc_language="English"):
    """Run the Sprint 2 pipeline (parser + NLP) on a single PDF, in memory.

    Equivalent to running parser.py then nlp_pipeline.py for one file, without
    needing it to already live in datasets/ or writing the intermediate
    output/<game>.json to disk first. doc_language is the language of the rulebook (see nlp_pipeline.analyse).
    """
    source = os.path.basename(pdf_path)
    pages = rulebook_parser.extract_pages(pdf_path)
    chunks = rulebook_parser.chunk_pages(pages)
    full_text = "\n\n".join(p["text"] for p in pages if p["text"])

    parsed = {
        "source": source,
        "pages": len(pages),
        "words": len(full_text.split()),
        "empty_pages": [p["page"] for p in pages if not p["text"]],
        "ocr_pages": [p["page"] for p in pages if p["ocr"]],
        "page_texts": pages,
        "chunks": chunks,
    }

    nlp = spacy.load("en_core_web_sm")
    known_words = nlp_pipeline.load_known_words() if doc_language == "English" else set()  # the word list is English
    tagged = nlp_pipeline.analyse(parsed, nlp, known_words, doc_language)
    return tagged


def run_chunk_detection(client, tagged, model=FINAL_MODEL, language_note=""):
    """Sprint 3 chunk-based pass: unclear wording, terminology, timing, local contradictions."""
    chunks = detect.build_chunks(tagged["sentences"])
    results, total_in, total_out = [], 0, 0
    for n, chunk in enumerate(chunks, start=1):
        text = detect.format_chunk(chunk, tagged["source"])
        try:
            issues, usage = detect.call_claude(client, detect.MODELS[model], text, language_note)
        except anthropic.APIError as e:
            print(f"  chunk {n}/{len(chunks)}: API error: {e}", file=sys.stderr)
            continue
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        for issue in issues:
            if issue.get("is_extraction_artifact"):
                continue  # drop text-extraction noise before it ever reaches the user
            ids, quote_found = detect.verify(issue, chunk)
            results.append({
                **issue,
                "sentence_ids": ids,
                "quote_verified": quote_found,
                "pages": sorted({s["page"] for s in chunk if s["id"] in ids}),
                "scope": "local",
            })
    return results, total_in, total_out


def run_contradiction_detection(client, tagged, model=FINAL_MODEL, language_note=""):
    """Sprint 3 whole-document pass: contradictions that may span distant pages."""
    document = detect_contradictions.format_document(tagged["sentences"], tagged["source"])
    try:
        items, usage = detect_contradictions.call_claude(
            client, detect.MODELS[model], document, language_note)
    except anthropic.APIError as e:
        print(f"  whole-document contradiction pass: API error: {e}", file=sys.stderr)
        return [], 0, 0

    results = []
    for item in items:
        ids, quote_found = detect.verify(item, tagged["sentences"])
        pages = sorted({s["page"] for s in tagged["sentences"] if s["id"] in ids})
        results.append({
            **item,
            "category": "contradiction",
            "sentence_ids": ids,
            "quote_verified": quote_found,
            "pages": pages,
            "scope": "document",
        })
    return results, usage.input_tokens, usage.output_tokens


def analyze(pdf_path, model=FINAL_MODEL, doc_language=None, report_language=language_support.DEFAULT_LANGUAGE):
    """Analyse one PDF with the model "haiku" (default) or "sonnet" (a key of detect.MODELS).

    doc_language is the language of the rulebook (a name or code; None or "auto" detects it from the PDF) and
    report_language the language of the explanations. The rulebook is analysed in its own language and is not
    translated; the quotes, the pages and the numbers of the sentences are those of the PDF. An English rulebook with
    an English report is analysed exactly as before.
    """
    if model not in detect.MODELS:
        raise ValueError(f"Unknown model '{model}'; choose one of {list(detect.MODELS)}")
    report_language = language_support.normalize_language(report_language)
    doc_language = resolve_document_language(pdf_path, doc_language)
    note = language_support.language_note(doc_language, report_language)
    print(f"Parsing and tagging: {pdf_path} (model {detect.MODELS[model]}, rulebook in {doc_language}, report in {report_language})")
    tagged = parse_and_tag(pdf_path, doc_language)
    print(f"  {tagged['stats']['pages']} pages, {tagged['stats']['sentences']} sentences "
          f"({tagged['stats']['noise_sentences']} noise)")

    client = anthropic.Anthropic()

    print("Running chunk-based detection (unclear wording, terminology, timing, local contradictions)...")
    local_issues, li_in, li_out = run_chunk_detection(client, tagged, model, note)
    print(f"  {len(local_issues)} issues found")

    print("Running whole-document contradiction detection...")
    contradictions, ci_in, ci_out = run_contradiction_detection(client, tagged, model, note)
    print(f"  {len(contradictions)} candidates found")

    all_issues = local_issues + contradictions
    for n, issue in enumerate(all_issues):
        issue["id"] = n

    result = {
        "source": tagged["source"],
        "model": detect.MODELS[model],
        "language": {"document": doc_language, "report": report_language},
        "analyzed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": tagged["stats"],
        "terminology": tagged["terminology"],
        "issues": all_issues,
        "tokens": {"input": li_in + ci_in, "output": li_out + ci_out},
    }

    os.makedirs(RESULT_DIR, exist_ok=True)
    stem = os.path.splitext(tagged["source"])[0]
    out_path = os.path.join(RESULT_DIR, f"{stem}.analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # The sentences, as extracted, are kept next to the analysis: the questions of a player (api.py, POST /ask) are
    # answered from them without reading the PDF again.
    with open(os.path.join(RESULT_DIR, f"{stem}.sentences.json"), "w", encoding="utf-8") as f:
        json.dump({"source": tagged["source"], "language": doc_language, "sentences": tagged["sentences"]}, f, ensure_ascii=False)

    by_cat = {}
    for i in all_issues:
        by_cat[i["category"]] = by_cat.get(i["category"], 0) + 1
    total_tokens = result["tokens"]
    p_in, p_out = detect.PRICES.get(model, (0, 0))
    cost = total_tokens["input"] / 1e6 * p_in + total_tokens["output"] / 1e6 * p_out
    print(f"\nTotal issues: {len(all_issues)} {by_cat}")
    print(f"Tokens: {total_tokens['input']} in / {total_tokens['output']} out (about ${cost:.4f})")
    print(f"Saved: {out_path}")
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="path to any rulebook PDF")
    group.add_argument("--game", help="name of a PDF already in datasets/ (partial match)")
    ap.add_argument("--doc-language", default="auto", help="language of the rulebook (default: detected from the PDF)")
    ap.add_argument("--report-language", default=language_support.DEFAULT_LANGUAGE, help="language of the explanations")
    args = ap.parse_args()

    if args.pdf:
        pdf_path = args.pdf
    else:
        matches = [f for f in os.listdir(rulebook_parser.DATASET_DIR)
                   if f.endswith(".pdf") and args.game.lower() in f.lower()]
        if len(matches) != 1:
            sys.exit(f"Expected one dataset PDF matching '{args.game}', found: {matches}")
        pdf_path = os.path.join(rulebook_parser.DATASET_DIR, matches[0])

    if not os.path.exists(pdf_path):
        sys.exit(f"File not found: {pdf_path}")

    analyze(pdf_path, doc_language=args.doc_language, report_language=args.report_language)


if __name__ == "__main__":
    main()
