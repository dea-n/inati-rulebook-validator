"""Sprint 3: LLM-based detection of rulebook issues with the Claude API.

Reads output/<game>.nlp.json (sentences from nlp_pipeline.py), sends the text to
Claude in chunks, and saves the detected issues to
output/<game>.<model>.detections.json.

Usage:
    python detect.py --game Carcassonne --model haiku
    python detect.py --game Carcassonne --model haiku --max-chunks 2 --hints
"""
import argparse
import json
import os
import sys
import time

import anthropic

import language as language_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

PROMPT_VERSION = "v2"
CHUNK_WORDS = 500

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-5",
}
# USD per million tokens (input, output). Check the Console pricing page before quoting these.
# Sonnet is listed at twice the price per token of Haiku (Chapter 6); it is only used when the user picks it.
PRICES = {"haiku": (1.0, 5.0), "sonnet": (2.0, 10.0)}

CATEGORIES = [
    "unclear_wording",
    "inconsistent_terminology",
    "contradiction",
    "unclear_timing",
]

SYSTEM_PROMPT = """You are an expert board game editor who proofreads rulebooks for players.

You receive a numbered excerpt of a rulebook. Find real problems that would make players
misunderstand a rule or argue about it during play. Report only these four categories:

- unclear_wording: a rule with more than one reasonable reading, or vague wording that
  changes how the game is played (for example, "some", "usually", an unclear "it" or "they").
- inconsistent_terminology: the same component, action or concept called different names,
  or one name used for different things, in a way that could confuse players.
- contradiction: two statements in the excerpt that cannot both be true. Cite both sentences.
- unclear_timing: the order or moment of events is not clear (when an effect happens, what
  happens first, whether something can be done at any time).

Guidelines:
- Be conservative. Do not report a sentence that is clear in context. It is better to report
  fewer real problems than many doubtful ones. Returning an empty list is a valid answer.
- The text was extracted from a PDF, and sometimes text from side boxes, captions, examples
  or card artwork is mixed into the middle of a sentence, so the sentence reads as scrambled
  or cut off. This is an extraction problem, not a rulebook problem. When the problem you
  see is only caused by that, set "is_extraction_artifact" to true and use category
  unclear_wording. Set it to false only for problems that exist in the rulebook's own
  wording. A sentence that is fragmented, has words missing, or seems to continue in the
  middle of another sentence is almost always an extraction artifact.
- Only use the excerpt. Do not use your own knowledge of the game's real rules.
- "quote" must be copied word for word from exactly ONE sentence in the excerpt: the
  sentence that contains the problem. Never join several sentences into one quote.
- "sentence_ids" are the numbers in [S..] of the sentences involved. The first id is the
  sentence you quote. A contradiction needs two or more ids.
- A contradiction means two clear statements that cannot both be true. A single confusing or
  ambiguous sentence is unclear_wording, not a contradiction.
- Ignore typos and grammar mistakes unless they change the meaning of a rule.
- "suggested_rewrite" is a clearer replacement wording that keeps the intended rule. Only
  rewrite the quoted sentence.
- "explanation" is one or two plain sentences saying why players could misread it.
- "severity": high = likely to change the outcome of a game, medium = likely to cause a
  pause or an argument, low = minor.

Report your findings by calling the report_issues tool."""

TOOL = {
    "name": "report_issues",
    "description": "Report the rulebook issues found in the excerpt (empty list if none).",
    "input_schema": {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string", "enum": CATEGORIES},
                        "sentence_ids": {"type": "array", "items": {"type": "integer"}},
                        "quote": {"type": "string"},
                        "explanation": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "suggested_rewrite": {"type": "string"},
                        "is_extraction_artifact": {"type": "boolean"},
                    },
                    "required": ["category", "sentence_ids", "quote", "explanation",
                                 "severity", "suggested_rewrite", "is_extraction_artifact"],
                },
            }
        },
        "required": ["issues"],
    },
}


REQUIRED_FIELDS = set(TOOL["input_schema"]["properties"]["issues"]["items"]["required"])


def build_chunks(sentences, max_words=CHUNK_WORDS):
    """Group consecutive clean sentences into chunks of about max_words words.

    Chunks are cut at section boundaries when they are already reasonably full,
    so a rule and its explanation stay together.
    """
    chunks, current, words = [], [], 0

    def flush():
        nonlocal current, words
        if current:
            chunks.append(current)
        current, words = [], 0

    for s in sentences:
        if s["noise"]:
            continue
        n = len(s["text"].split())
        section_changed = current and s["section"] != current[-1]["section"]
        if current and (words + n > max_words or (section_changed and words > 0.6 * max_words)):
            flush()
        current.append(s)
        words += n
    flush()
    return chunks


def format_chunk(chunk, game, candidates_by_sentence=None):
    lines = [f"Rulebook: {game}", ""]
    last_section = None
    for s in chunk:
        if s["section"] != last_section:
            lines.append(f"## Section: {s['section']} (page {s['page']})")
            last_section = s["section"]
        line = f"[S{s['id']}] {s['text']}"
        if candidates_by_sentence and s["id"] in candidates_by_sentence:
            line += "   <automatic pre-check: " + ", ".join(candidates_by_sentence[s["id"]]) + ">"
        lines.append(line)
    return "\n".join(lines)


def call_claude(client, model_id, user_text, language_note=""):
    """One API call with forced structured output. Returns (issues, usage).

    language_note (see language.language_note) tells the model the language of the rulebook and the language of the
    explanations. It is empty for an English rulebook with an English report, so that the prompt is then exactly the
    one that was evaluated."""
    response = client.messages.create(
        model=model_id,
        max_tokens=4000,
        system=language_support.with_note(SYSTEM_PROMPT, language_note),
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "report_issues"},
        messages=[{"role": "user", "content": user_text}],
    )
    for block in response.content:
        if block.type == "tool_use":
            issues = block.input.get("issues", [])
            if isinstance(issues, str):  # occasionally the list comes back as a JSON string
                try:
                    issues = json.loads(issues)
                except json.JSONDecodeError:
                    issues = []
            return [i for i in issues if isinstance(i, dict) and REQUIRED_FIELDS <= i.keys()], response.usage
    return [], response.usage


def verify(issue, chunk):
    """Check that the sentence ids exist in the chunk and the quote is in a cited sentence."""
    by_id = {s["id"]: s["text"] for s in chunk}
    ids = [i for i in issue["sentence_ids"] if i in by_id]
    quote = " ".join(issue["quote"].split()).lower()
    quote_found = any(quote and quote in " ".join(by_id[i].split()).lower() for i in ids)
    return ids, quote_found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--game", required=True, help="part of the rulebook file name, e.g. Carcassonne")
    parser.add_argument("--model", choices=MODELS, default="haiku")
    parser.add_argument("--max-chunks", type=int, default=None, help="only analyse the first N chunks")
    parser.add_argument("--hints", action="store_true",
                        help="show the rule-based pre-check flags to the model")
    args = parser.parse_args()

    files = [f for f in os.listdir(OUTPUT_DIR)
             if f.endswith(".nlp.json") and args.game.lower() in f.lower()]
    if len(files) != 1:
        sys.exit(f"Expected one rulebook matching '{args.game}', found: {files}")
    with open(os.path.join(OUTPUT_DIR, files[0]), encoding="utf-8") as f:
        data = json.load(f)

    candidates = {}
    if args.hints:
        for i in data["issues"]:
            if i["sentence_id"] is not None and i["category"] != "timing_cue":
                candidates.setdefault(i["sentence_id"], []).append(i["category"])

    chunks = build_chunks(data["sentences"])
    if args.max_chunks:
        chunks = chunks[:args.max_chunks]
    print(f"{data['source']}: {len(chunks)} chunks, model {MODELS[args.model]}, "
          f"hints {'on' if args.hints else 'off'}")

    client = anthropic.Anthropic()
    results, total_in, total_out = [], 0, 0
    for n, chunk in enumerate(chunks, start=1):
        text = format_chunk(chunk, data["source"], candidates if args.hints else None)
        try:
            issues, usage = call_claude(client, MODELS[args.model], text)
        except anthropic.APIError as e:
            print(f"  chunk {n}/{len(chunks)}: API error: {e}")
            continue
        total_in += usage.input_tokens
        total_out += usage.output_tokens
        for issue in issues:
            ids, quote_found = verify(issue, chunk)
            results.append({
                **issue,
                "sentence_ids": ids,
                "quote_verified": quote_found,
                "pages": sorted({s["page"] for s in chunk if s["id"] in ids}),
                "chunk": n,
            })
        print(f"  chunk {n}/{len(chunks)} (pages {chunk[0]['page']}-{chunk[-1]['page']}): "
              f"{len(issues)} issues")

    out = {
        "source": data["source"],
        "model": MODELS[args.model],
        "prompt_version": PROMPT_VERSION,
        "hints": args.hints,
        "chunks_analysed": len(chunks),
        "tokens": {"input": total_in, "output": total_out},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "issues": results,
    }
    stem = files[0].replace(".nlp.json", "")
    suffix = f"{args.model}.{PROMPT_VERSION}{'.hints' if args.hints else ''}"
    out_path = os.path.join(OUTPUT_DIR, f"{stem}.{suffix}.detections.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    artifacts = [r for r in results if r.get("is_extraction_artifact")]
    real = [r for r in results if not r.get("is_extraction_artifact")]
    by_cat = {}
    for r in real:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    unverified = sum(1 for r in real if not r["quote_verified"])
    print(f"\nRulebook issues: {len(real)} {by_cat}")
    print(f"Flagged as extraction artifacts (ignored): {len(artifacts)}")
    print(f"Rulebook issues whose quote is not in one cited sentence: {unverified}")
    print(f"Tokens: {total_in} in / {total_out} out", end="")
    if args.model in PRICES:
        p_in, p_out = PRICES[args.model]
        print(f"  (about ${total_in / 1e6 * p_in + total_out / 1e6 * p_out:.4f})")
    else:
        print("  (check cost on the Console usage page)")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
