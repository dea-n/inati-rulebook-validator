"""Summaries behind Chapter 6 (Evaluation): which kinds of real problems were found, how the contradiction
candidates were judged, what the explanations of the deployed findings look like, and what a real finding cost.

Everything is read from files that already exist in output/ (labelling spreadsheets, *.contradictions.json,
output/analysis/*.analysis.json and the saved token counts). Nothing here calls the paid API.

Usage (from the src folder):
    python evaluation_summary.py
"""
import collections
import difflib
import glob
import json
import os
import re
import statistics

import openpyxl

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE_DIR, "output")
ANALYSIS = os.path.join(OUT, "analysis")
PRICE_IN, PRICE_OUT = 1.0, 5.0  # USD per million tokens, Claude Haiku 4.5

SENTENCE_REF = re.compile(r"\bS\d{1,4}\b|\b[Ss]entences?\s+\d{1,4}")  # internal references such as "S301" or "sentence 61"


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- 1. real findings by category (labelling spreadsheets)
def real_by_category():
    """Count the spreadsheet rows with the verdict 'real' by the category of the first run that reported them."""
    counts, rows = collections.Counter(), []
    for path in sorted(glob.glob(os.path.join(OUT, "labeling", "*.xlsx"))):
        if os.path.basename(path).startswith("~$"):
            continue  # the lock file of a sheet that is open in Excel
        sheet = openpyxl.load_workbook(path, data_only=True)["Findings"]
        values = list(sheet.iter_rows(values_only=True))
        col = {h: i for i, h in enumerate(values[0])}
        for r in values[1:]:
            if str(r[col["YOUR VERDICT"]]).strip() != "real":
                continue
            category = re.search(r":\s*([a-z_]+)\s*\(", str(r[col["category (per run)"]])).group(1)
            category = {"unclear_terminology": "inconsistent_terminology"}.get(category, category)
            counts[category] += 1
            rows.append((os.path.basename(path)[:14], r[col["page"]], category, str(r[col["sentence"]]).strip()))
    return counts, rows


# ---------------------------------------------------------------- 2. verdicts of the contradiction candidates
def contradiction_verdicts():
    """The whole-document candidates of both models, with the verdict and note recorded in the JSON files."""
    items = []
    for path in sorted(glob.glob(os.path.join(OUT, "*.contradictions.json"))):
        model = "sonnet" if ".sonnet." in path else "haiku"
        for n, issue in enumerate(load_json(path)["issues"]):
            items.append(dict(file=os.path.basename(path), model=model, n=n, pages=issue["pages"],
                              verdict=issue.get("human_verdict"), note=issue.get("human_note", "")))
    return collections.Counter(i["verdict"] for i in items), items


# ---------------------------------------------------------------- 3. what the explanations look like
def words(text):
    return re.findall(r"[A-Za-z0-9’'\-]+", text)


def changed_words(before, after):
    """Number of words that differ between two texts (a rewrite that only adds a comma gives 0)."""
    a, b = words(before.lower()), words(after.lower())
    matcher = difflib.SequenceMatcher(None, a, b)
    return sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal")


def syllables(word):
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    n = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and n > 1:
        n -= 1
    return max(n, 1)


def reading_ease(text):
    """Approximate Flesch reading ease (a vowel-group syllable count, so the value is only indicative)."""
    sentences = max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))
    tokens = re.findall(r"[A-Za-z’'\-]+", text)
    if not tokens:
        return None
    syl = sum(syllables(t) for t in tokens)
    return 206.835 - 1.015 * len(tokens) / sentences - 84.6 * syl / len(tokens)


def explanation_stats():
    findings = []
    for path in sorted(glob.glob(os.path.join(ANALYSIS, "*.analysis.json"))):
        findings += load_json(path)["issues"]
    n = len(findings)
    lengths = [len(words(f["explanation"])) for f in findings]
    ease = [reading_ease(f["explanation"]) for f in findings]
    with_ref = sum(1 for f in findings if SENTENCE_REF.search(f["explanation"]))
    non_contra = [f for f in findings if f["category"] != "contradiction"]
    changes = [changed_words(f["quote"], f["suggested_rewrite"]) for f in non_contra]
    sev = collections.Counter(f["severity"] for f in findings)
    q = statistics.quantiles(lengths, n=4)
    return dict(
        n=n, with_rewrite=sum(1 for f in findings if f.get("suggested_rewrite", "").strip()),
        with_quote=sum(1 for f in findings if f.get("quote", "").strip()), verified=sum(1 for f in findings if f.get("quote_verified")),
        median_words=statistics.median(lengths), q1_words=q[0], q3_words=q[2], min_words=min(lengths), max_words=max(lengths),
        median_ease=statistics.median(ease), with_sentence_ref=with_ref,
        n_non_contra=len(non_contra), small_rewrites=sum(1 for c in changes if c <= 2), identical_rewrites=sum(1 for c in changes if c == 0),
        median_changed=statistics.median(changes), severity=dict(sev),
    )


# ---------------------------------------------------------------- 4. cost of a confirmed real finding
def cost_per_real():
    """Cost of the two analysis passes (saved token counts, Haiku prices) divided by the findings judged real."""
    out = {}
    for name, stem, real_chunk in (("Carcassonne", "Carcassonne_20th_Anniversary_Rulebook", 4), ("Terraforming Mars", "TerraformingMars_Rulebook", 7)):
        chunk = load_json(os.path.join(OUT, f"{stem}.haiku.v2.detections.json"))["tokens"]
        doc = load_json(os.path.join(OUT, f"{stem}.haiku.contradictions.json"))
        real_doc = sum(1 for i in doc["issues"] if i.get("human_verdict") == "real")
        cost = sum(t["input"] / 1e6 * PRICE_IN + t["output"] / 1e6 * PRICE_OUT for t in (chunk, doc["tokens"]))
        out[name] = dict(cost=cost, real=real_chunk + real_doc, per_real=cost / (real_chunk + real_doc))
    return out


def main():
    counts, rows = real_by_category()
    print("Findings labelled real, by category (spreadsheets):", dict(counts), "total", sum(counts.values()))
    for r in rows:
        print("   ", r[0], "p" + str(r[1]), r[2], "|", r[3][:90])
    verdicts, items = contradiction_verdicts()
    print("\nContradiction candidates (whole-document pass, both models):", dict(verdicts), "total", len(items))
    for i in items:
        print("   ", i["model"], i["file"][:20], "#" + str(i["n"]), i["pages"], i["verdict"], "|", i["note"][:110])
    print("\nExplanations of the deployed findings:")
    for k, v in explanation_stats().items():
        print("   ", k, "=", round(v, 1) if isinstance(v, float) else v)
    print("\nCost per confirmed real finding (both passes, Haiku):")
    for k, v in cost_per_real().items():
        print("   ", k, {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()})


if __name__ == "__main__":
    main()
