"""Sprint 4 / O4: generate FAQ entries for flagged rulebook issues.

Takes the output of analyze_rulebook.py (a game's found issues, each already
carrying a suggested_rewrite from the detection stage) and asks Claude to turn
each one into a short, player-facing FAQ entry: a natural question a player
would actually ask, plus a clear ruling that resolves it.

This is a single batched call per rulebook (cheap), not one call per issue.

Usage:
    python generate_faq.py --game Carcassonne
"""
import argparse
import json
import os
import re
import sys

import anthropic

import detect_contradictions
import language as language_support
from detect import MODELS, PRICES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANALYSIS_DIR = os.path.join(BASE_DIR, "output", "analysis")

FINAL_MODEL = "haiku"  # same choice as analyze_rulebook.py, see its docstring

SYSTEM_PROMPT = """You write FAQ entries for board game rulebooks, in the style publishers
use to answer player questions (e.g. a "Rules Questions" or "FAQ" appendix).

You receive a numbered list of issues already found in a rulebook (unclear wording,
inconsistent terminology, a contradiction, or unclear timing), each with an explanation and a
suggested rewrite. For each issue, write one FAQ entry:

- "issue_number": the [N] number of the issue this entry answers, copied exactly. Every issue
  must get exactly one entry, even if you are not fully confident - use low confidence instead
  of skipping one.
- "question": a short, natural question a player would actually ask at the table because of
  this issue. Write it the way a real player types it into a forum, not a formal restatement
  of the problem (e.g. "Do I get a Terraform Rating point for the greenery tile itself, or only
  if it also raises oxygen?", not "What does the ambiguous TR clause mean?").
- "answer": a clear, direct ruling in 1-3 sentences that resolves the question. Base it on the
  suggested_rewrite and explanation you were given, and state the ruling as a designer would:
  confidently, with the reasoning only if it helps a player remember it correctly.
- "confidence": "high" if the rewrite/explanation given clearly implies a single correct
  ruling, "low" if the original issue was genuinely ambiguous enough that this is your best
  reasonable interpretation rather than a certain answer - in that case the answer should say
  so plainly (e.g. "Most likely... but the rulebook does not state this explicitly").
- If an issue labelled "contradiction" turns out, on closer reading, to not actually be a real
  conflict (both statements are consistent, e.g. they describe different things that happen to
  share a similar number), do not invent a ruling that favours one side. Instead set confidence
  to "low" and say plainly that the two statements are not actually in conflict, explaining why.

Report your entries by calling the write_faq tool, one entry per issue."""

TOOL = {
    "name": "write_faq",
    "description": "Write one FAQ entry per issue provided.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "issue_number": {"type": "integer"},
                        "question": {"type": "string"},
                        "answer": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "low"]},
                    },
                    "required": ["issue_number", "question", "answer", "confidence"],
                },
            }
        },
        "required": ["entries"],
    },
}


def format_issues(issues):
    lines = []
    for n, issue in enumerate(issues, start=1):
        lines.append(
            f"[{n}] category: {issue['category']} (pages {issue['pages']})\n"
            f"    quote: {issue['quote']}\n"
            f"    explanation: {issue['explanation']}\n"
            f"    suggested_rewrite: {issue['suggested_rewrite']}"
        )
    return "\n\n".join(lines)


def generate_faq_entries(client, all_issues, quiet=False, model=FINAL_MODEL, doc_language=None, report_language=None):
    """Generate FAQ entries for a list of issues (as produced by analyze_rulebook.py).

    model is a key of detect.MODELS: api.py passes the model the user chose for the analysis.
    doc_language and report_language: the questions and answers are written in the report language; with an English
    rulebook and an English report (the default) the prompt is the original one.

    Returns (faq, review_queue, usage). Shared by the CLI below and api.py, so the
    contradiction-routing safeguard (see module docstring / Chapter 3.5.4) and the
    issue-number-based pairing fix live in exactly one place.

    Sprint 3's evaluation measured only 6% precision (1/16) on "contradiction"
    findings (see output/evaluation/recall_evaluation.md): most turned out to be
    consistent statements the model misread as conflicting. Worse, testing this
    FAQ generator showed that even when explicitly warned, the model can invent a
    confident-sounding ruling for a false-positive contradiction that misstates the
    rulebook's own numbers. Given that measured unreliability, contradictions are
    routed to human review instead of an auto-generated answer.
    """
    issues = [i for i in all_issues if i["category"] != "contradiction"]
    needs_review = [i for i in all_issues if i["category"] == "contradiction"]
    review_queue = [{
        "issue_id": i["id"], "category": i["category"], "pages": i["pages"],
        "severity": i.get("severity"), "quote": i["quote"], "explanation": i["explanation"],
        "reason": "contradiction findings had only 6% measured precision - needs human confirmation before use",
    } for i in needs_review]

    if not issues:
        if not quiet:
            print(f"Only contradiction findings present - all {len(needs_review)} "
                  f"routed to human review, no FAQ entries generated.")
        return [], review_queue, None

    response = client.messages.create(
        model=MODELS[model],
        max_tokens=4000,
        system=language_support.with_note(SYSTEM_PROMPT, language_support.language_note(doc_language, report_language, "faq")),
        tools=[TOOL],
        tool_choice={"type": "tool", "name": "write_faq"},
        messages=[{"role": "user", "content": format_issues(issues)}],
    )
    entries = []
    for block in response.content:
        if block.type == "tool_use":
            entries = block.input.get("entries", [])

    # Match by the issue_number the model reported, not by list position, so a
    # skipped or reordered entry can never get silently paired with the wrong issue.
    by_number = {e["issue_number"]: e for e in entries if "issue_number" in e}
    faq, missing = [], []
    for n, issue in enumerate(issues, start=1):
        entry = by_number.get(n)
        if entry is None:
            missing.append(n)
            continue
        faq.append({
            "issue_id": issue["id"],
            "category": issue["category"],
            "pages": issue["pages"],
            "severity": issue.get("severity"),
            "question": entry["question"],
            "answer": entry["answer"],
            "confidence": entry["confidence"],
            "source_quote": issue["quote"],
        })
    if missing and not quiet:
        print(f"Warning: the model did not return an entry for issue(s) {missing} "
              f"(quote: {[issues[n-1]['quote'][:60] for n in missing]}) - skipped, not guessed.",
              file=sys.stderr)

    return faq, review_queue, response.usage


# --- A player's own question about a rulebook (any language), answered from the rulebook only ----------------------

ASK_SYSTEM_PROMPT = """You answer a player's question about a board game rulebook, using ONLY the rulebook text that you receive.

The rulebook is given as numbered sentences, grouped by section and page. The question can be in any language and the
rulebook can be in another one. Write the answer in the language of the question.

Rules:
- Use only the sentences of the rulebook. Do not use your own knowledge of the game or of similar games.
- If the rulebook states the answer, set "answerable" to true, answer in one to four sentences, and list in
  "sentence_ids" every sentence that supports the answer (the numbers in [S..]).
- If the rulebook does not give enough information, or only part of it, set "answerable" to false. Say in the answer
  what the rulebook does say about the topic (if anything) and that it does not settle the question. Do not guess and
  do not invent a rule.
- If the text looks damaged (scrambled or cut off) where the answer should be, say so instead of guessing.
- If the rulebook is unclear or contradicts itself on the point, say that, and cite the sentences.
- The player cannot see the sentence numbers. Never mention them in the answer, in any form ("[S12]", "sentence 12",
  "Satz 12"); they go only in "sentence_ids". Refer to the rule itself, not to its number.

Report your answer by calling the answer_question tool."""

ASK_TOOL = {
    "name": "answer_question",
    "description": "Answer the question from the rulebook, or say that the rulebook does not answer it.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answerable": {"type": "boolean"},
            "answer": {"type": "string"},
            "sentence_ids": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["answerable", "answer", "sentence_ids"],
    },
}

NO_SUPPORT = "The rulebook text does not support an answer to this question."
# The numbers of the sentences are internal. A model sometimes writes them into the answer ("[S42]", "(S42, S43)") in spite of
# the prompt; they mean nothing to a player, so they are removed. The pages and the quotes come from "sentence_ids".
SENTENCE_REFERENCE = re.compile(r"\s*[\[(]\s*S\d+(?:\s*[,;\-–]\s*S?\d+)*\s*[\])]")


def answer_question(client, sentences, source, question, model=FINAL_MODEL):
    """Answer a question from the sentences of a rulebook. Returns a dict with answerable, answer, sentence_ids,
    pages, quotes and tokens.

    The model must cite sentences. The ids are checked against the rulebook, and the pages and the quotes are taken from
    the rulebook, not from the model. An answer without a valid citation is turned into "not answerable", so that an
    answer is never given without support in the text.
    """
    document = detect_contradictions.format_document(sentences, source)
    response = client.messages.create(
        model=MODELS[model],
        max_tokens=1000,
        system=ASK_SYSTEM_PROMPT,
        tools=[ASK_TOOL],
        tool_choice={"type": "tool", "name": "answer_question"},
        messages=[{"role": "user", "content": f"{document}\n\nQuestion: {question}"}],
    )
    data = {}
    for block in response.content:
        if block.type == "tool_use":
            data = block.input
    by_id = {s["id"]: s for s in sentences if not s["noise"]}
    ids = [i for i in dict.fromkeys(data.get("sentence_ids") or []) if i in by_id]
    claimed = bool(data.get("answerable"))
    answer = SENTENCE_REFERENCE.sub("", str(data.get("answer") or "")).strip()
    answerable = claimed and bool(ids) and bool(answer)
    if claimed and not answerable or not answer:
        answer = NO_SUPPORT  # an answer that the text does not support is never shown
    return {
        "answerable": answerable,
        "answer": answer,
        "sentence_ids": ids,
        "pages": sorted({by_id[i]["page"] for i in ids}),
        "quotes": [by_id[i]["text"] for i in ids][:4],
        "tokens": {"input": response.usage.input_tokens, "output": response.usage.output_tokens},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--game", required=True, help="part of the rulebook file name, e.g. Carcassonne")
    args = ap.parse_args()

    files = [f for f in os.listdir(ANALYSIS_DIR)
             if f.endswith(".analysis.json") and args.game.lower() in f.lower()]
    if len(files) != 1:
        sys.exit(f"Expected one analysis file matching '{args.game}' in {ANALYSIS_DIR}, "
                  f"found: {files}. Run analyze_rulebook.py first.")
    with open(os.path.join(ANALYSIS_DIR, files[0]), encoding="utf-8") as f:
        analysis = json.load(f)

    all_issues = analysis["issues"]
    if not all_issues:
        print(f"{analysis['source']}: no issues to write FAQ entries for.")
        return

    n_contra = sum(1 for i in all_issues if i["category"] == "contradiction")
    print(f"{analysis['source']}: writing FAQ entries for {len(all_issues) - n_contra} issue(s) "
          f"({n_contra} contradiction finding(s) routed to human review instead)...")

    client = anthropic.Anthropic()
    faq, review_queue, usage = generate_faq_entries(client, all_issues)

    out_path = os.path.join(ANALYSIS_DIR, files[0].replace(".analysis.json", ".faq.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "source": analysis["source"],
            "model": MODELS[FINAL_MODEL],
            "faq": faq,
            "needs_human_review": review_queue,
        }, f, ensure_ascii=False, indent=2)

    if usage:
        p_in, p_out = PRICES.get(FINAL_MODEL, (0, 0))
        cost = usage.input_tokens / 1e6 * p_in + usage.output_tokens / 1e6 * p_out
        print(f"Wrote {len(faq)} FAQ entries. Tokens: {usage.input_tokens} in / "
              f"{usage.output_tokens} out (about ${cost:.4f})")
    print(f"Saved: {out_path}\n")
    for e in faq:
        print(f"Q: {e['question']}")
        print(f"A: {e['answer']}  [{e['confidence']} confidence]")
        print()


if __name__ == "__main__":
    main()
