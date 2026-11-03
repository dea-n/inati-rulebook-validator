"""Sprint 3: whole-document contradiction detection.

detect.py only compares sentences within a single ~500-word chunk, so it
structurally cannot catch a contradiction between, say, a rule on page 3 and
a rule on page 12. This script instead sends the WHOLE cleaned rulebook to
Claude in one call (all four rulebooks are well under 9,000 words, far inside
the model's context window) and asks only for contradictions between
statements that may be anywhere in the document.

Saves output/<game>.<model>.contradictions.json, in the same issue shape as
detect.py so it can be merged with make_labeling_sheet.py / score_precision.py.

Usage:
    python detect_contradictions.py --game Carcassonne --model haiku
"""
import argparse
import json
import os
import sys
import time

import anthropic

import language as language_support
from detect import BASE_DIR, OUTPUT_DIR, MODELS, PRICES, verify  # reuse, don't duplicate

PROMPT_VERSION = "contradictions-v1"

SYSTEM_PROMPT = """You are an expert board game editor checking a rulebook for internal contradictions.

You receive the full text of a rulebook, as numbered sentences grouped by section and page.
Find pairs (or small groups) of sentences that CANNOT both be true - genuine contradictions -
anywhere in the document, including cases where the two statements are far apart (different
pages or sections). This is different from a single confusing sentence: you need at least two
separate statements that conflict with each other.

Examples of what counts as a contradiction:
- Two different values given for the same thing (a cost, a starting amount, a limit, a VP total).
- A rule stated one way in a summary/quick-reference section and differently in the main rules.
- A general rule stated, then an exception or later sentence that silently reverses it rather
  than qualifying it.
- The same named component or track described with two different behaviours.

What does NOT count as a contradiction (do not report these):
- A single ambiguous or vague sentence with no second conflicting statement.
- Two statements that are both true but easy to misread as conflicting once you read carefully
  (e.g. one is clearly a special case or exception of the other, or an example illustrating the
  general rule with specific numbers).
- Two mentions of the same rule using different wording but the same meaning.
- Anything caused by broken PDF text extraction: sentences that are fragmented, cut off mid-word,
  mix unrelated topics together, or contain stray card/component names and numbers scattered
  through them. These are extraction noise, not rulebook contradictions - if the apparent
  conflict only exists because the text is garbled, do not report it at all (do not report it
  as a contradiction with is_extraction_artifact=true either; simply skip it, since without
  reliable text there is nothing to genuinely compare).
- Your own outside knowledge of how the real game is actually played. Judge only the text given.

Be conservative: a real contradiction is a strong, citable claim. If you are not confident both
statements are clearly stated and clearly conflicting, do not report it. Returning an empty list
is a valid and expected answer for a well-written rulebook.

For each contradiction:
- "sentence_ids" must list every sentence involved (two or more), in the order that makes the
  conflict clearest. The first id is the one to quote.
- "quote" is copied word for word from exactly the sentence at the first sentence_id.
- "explanation" must state plainly what the first statement says, what the other statement(s)
  say, and why they cannot both be true. Mention the page numbers involved.
- "suggested_rewrite" proposes how the rulebook could resolve the conflict (e.g. which statement
  is likely correct, or how to add a qualifying clause), not just a rewrite of one sentence.
- "severity": high = likely to change the outcome of a game, medium = likely to cause a pause or
  an argument, low = minor / unlikely to matter in practice.

Report your findings by calling the report_contradictions tool."""

TOOL = {
    "name": "report_contradictions",
    "description": "Report genuine cross-document contradictions found in the rulebook (empty list if none).",
    "input_schema": {
        "type": "object",
        "properties": {
            "contradictions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence_ids": {"type": "array", "items": {"type": "integer"}, "minItems": 2},
                        "quote": {"type": "string"},
                        "explanation": {"type": "string"},
                        "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                        "suggested_rewrite": {"type": "string"},
                    },
                    "required": ["sentence_ids", "quote", "explanation", "severity", "suggested_rewrite"],
                },
            }
        },
        "required": ["contradictions"],
    },
}

REQUIRED_FIELDS = set(TOOL["input_schema"]["properties"]["contradictions"]["items"]["required"])


def format_document(sentences, game):
    """Render every non-noise sentence, grouped by section, with page numbers."""
    lines = [f"Rulebook: {game}", ""]
    last_section = None
    for s in sentences:
        if s["noise"]:
            continue
        if s["section"] != last_section:
            lines.append(f"\n## Section: {s['section']} (page {s['page']})")
            last_section = s["section"]
        lines.append(f"[S{s['id']}] {s['text']}")
    return "\n".join(lines)


def call_claude(client, model_id, user_text, language_note=""):
    """The whole-document call; language_note is added to the system prompt as in detect.call_claude."""
    response = client.messages.create(
        model=model_id,
        max_tokens=4000,
        system=language_support.with_note(SYSTEM_PROMPT, language_note),
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "report_contradictions"},
        messages=[{"role": "user", "content": user_text}],
    )
    for block in response.content:
        if block.type == "tool_use":
            items = block.input.get("contradictions", [])
            if isinstance(items, str):
                try:
                    items = json.loads(items)
                except json.JSONDecodeError:
                    items = []
            return [i for i in items if isinstance(i, dict) and REQUIRED_FIELDS <= i.keys()], response.usage
    return [], response.usage


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--game", required=True, help="part of the rulebook file name, e.g. Carcassonne")
    parser.add_argument("--model", choices=MODELS, default="haiku")
    args = parser.parse_args()

    files = [f for f in os.listdir(OUTPUT_DIR)
              if f.endswith(".nlp.json") and args.game.lower() in f.lower()]
    if len(files) != 1:
        sys.exit(f"Expected one rulebook matching '{args.game}', found: {files}")
    with open(os.path.join(OUTPUT_DIR, files[0]), encoding="utf-8") as f:
        data = json.load(f)

    document = format_document(data["sentences"], data["source"])
    words = len(document.split())
    print(f"{data['source']}: whole document, {words} words, model {MODELS[args.model]}")

    client = anthropic.Anthropic()
    try:
        items, usage = call_claude(client, MODELS[args.model], document)
    except anthropic.APIError as e:
        sys.exit(f"API error: {e}")

    results = []
    for item in items:
        ids, quote_found = verify(item, data["sentences"])
        pages = sorted({s["page"] for s in data["sentences"] if s["id"] in ids})
        results.append({
            **item,
            "category": "contradiction",
            "is_extraction_artifact": False,
            "sentence_ids": ids,
            "quote_verified": quote_found,
            "pages": pages,
            "cross_page": len(pages) > 1,
        })

    out = {
        "source": data["source"],
        "model": MODELS[args.model],
        "prompt_version": PROMPT_VERSION,
        "document_words": words,
        "tokens": {"input": usage.input_tokens, "output": usage.output_tokens},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "issues": results,
    }
    stem = files[0].replace(".nlp.json", "")
    out_path = os.path.join(OUTPUT_DIR, f"{stem}.{args.model}.contradictions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    cross_page = sum(1 for r in results if r["cross_page"])
    unverified = sum(1 for r in results if not r["quote_verified"])
    print(f"Contradictions found: {len(results)} ({cross_page} spanning more than one page)")
    print(f"Quotes not found in the cited sentence: {unverified}")
    print(f"Tokens: {usage.input_tokens} in / {usage.output_tokens} out", end="")
    if args.model in PRICES:
        p_in, p_out = PRICES[args.model]
        print(f"  (about ${usage.input_tokens / 1e6 * p_in + usage.output_tokens / 1e6 * p_out:.4f})")
    else:
        print("  (check cost on the Console usage page)")
    for r in results:
        print(f"\n  [{r['severity']}] pages {r['pages']} sentences {r['sentence_ids']}")
        print(f"  quote: {r['quote'][:150]}")
        print(f"  why:   {r['explanation'][:220]}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
