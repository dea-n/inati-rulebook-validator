"""Sprint 3 evaluation: precision of each detection run against a labeled sheet.

Reads the human verdicts from output/labeling/<game>.labeling.xlsx and checks
each raw detection run (output/<game>.<model>.detections.json) against them.

Precision here is measured only over the issues a run would actually surface to
a user (is_extraction_artifact == False from the model itself). A human verdict
of "real" counts as correct; "not_real", "extraction_artifact" or "unsure" count
as incorrect (unsure is reported separately too, since it is not a firm miss).

Usage:
    python score_precision.py --game Carcassonne
"""
import argparse
import glob
import json
import os
import sys

from openpyxl import load_workbook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labeling")

VALID_VERDICTS = {"real", "not_real", "extraction_artifact", "unsure"}


def load_sentence_texts(stem):
    """Map each sentence's id to its exact text, from the nlp_pipeline output.

    Used to recover a detection issue's actual source sentence (via its
    sentence_ids), so it can be matched against the labeling sheet by that
    sentence's exact text rather than by the id alone. Matching by text (not
    id) is deliberate: some rulebooks (e.g. Catan's icon-heavy tile summary)
    repeat an identical short sentence at several different ids, and a human
    verdict on one occurrence of that exact text applies equally to the
    others - matching by id alone would silently miss those duplicates.
    """
    with open(os.path.join(OUTPUT_DIR, f"{stem}.nlp.json"), encoding="utf-8") as f:
        nlp = json.load(f)
    return {s["id"]: s["text"].strip() for s in nlp["sentences"]}


def load_verdicts(path):
    wb = load_workbook(path, data_only=True)
    ws = wb["Findings"]
    by_text = {}
    empty = 0
    for row in ws.iter_rows(min_row=2, values_only=True):
        rid, page, section, sentence, context, found_by = row[:6]
        verdict, notes = row[10], row[11]
        if sentence is None:
            continue
        v = (str(verdict).strip().lower() if verdict else "")
        if v not in VALID_VERDICTS:
            empty += 1
            continue
        by_text[str(sentence).strip()] = v
    return by_text, empty


def run_label(data):
    model = "sonnet" if "sonnet" in data["model"] else "haiku"
    return f"{model}{'+hints' if data['hints'] else ''}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    args = parser.parse_args()

    label_files = sorted(f for f in glob.glob(os.path.join(LABEL_DIR, "*.labeling.xlsx"))
                         if args.game.lower() in os.path.basename(f).lower()
                         and not os.path.basename(f).startswith("~$"))  # "~$" files are the lock files of an open Excel
    if not label_files:
        sys.exit(f"No labeling sheet matching '{args.game}' found")
    # a sheet built with make_labeling_sheet.py --model <name> has that model's name
    # appended to the rulebook stem; split it back off to find the source files
    stems, filters = set(), set()
    for path in label_files:
        sheet_stem = os.path.basename(path).replace(".labeling.xlsx", "")
        model_filter, stem = None, sheet_stem
        for m in ("haiku", "sonnet"):
            if sheet_stem.endswith(f".{m}"):
                model_filter, stem = m, sheet_stem[: -len(m) - 1]
                break
        stems.add(stem)
        filters.add(model_filter)
    if len(stems) != 1:
        sys.exit(f"Expected the labeling sheets of one rulebook for '{args.game}', found {label_files}")
    stem = stems.pop()
    # One sheet built for one model scores only that model's runs. Several sheets of the same rulebook (for example the
    # first one for Sonnet and a later one for the Haiku findings that were still without a verdict) are read together
    # and score every run.
    model_filter = filters.pop() if len(label_files) == 1 else None

    id_to_text = load_sentence_texts(stem)
    verdicts, empty = {}, 0
    for path in label_files:
        sheet_verdicts, sheet_empty = load_verdicts(path)
        verdicts.update(sheet_verdicts)
        empty += sheet_empty
    if empty:
        print(f"Note: {empty} row(s) in the labeling sheet(s) have no valid verdict yet and are skipped.")

    run_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{stem}.*.detections.json")))
    if model_filter:
        run_files = [f for f in run_files if f".{model_filter}." in os.path.basename(f)]
    if not run_files:
        sys.exit("No detection runs found for this game.")

    print(f"{'Run':16}{'Surfaced':>9}{'Real':>6}{'NotReal':>8}{'Artifact':>9}{'Unsure':>7}{'Precision':>10}")
    rows = []
    for path in run_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        label = run_label(data)
        counts = {"real": 0, "not_real": 0, "extraction_artifact": 0, "unsure": 0}
        unmatched = []
        for issue in data["issues"]:
            if issue.get("is_extraction_artifact"):
                continue  # the run itself would not surface this one
            sid = issue["sentence_ids"][0] if issue["sentence_ids"] else None
            source_text = id_to_text.get(sid) if sid is not None else None
            v = verdicts.get(source_text) if source_text is not None else None
            if v is None:
                unmatched.append(issue["quote"])
                continue
            counts[v] += 1
        surfaced = sum(counts.values())
        precision = counts["real"] / surfaced if surfaced else float("nan")
        rows.append((label, surfaced, counts, precision, unmatched))
        print(f"{label:16}{surfaced:>9}{counts['real']:>6}{counts['not_real']:>8}"
              f"{counts['extraction_artifact']:>9}{counts['unsure']:>7}{precision:>10.0%}")

    for label, surfaced, counts, precision, unmatched in rows:
        if unmatched:
            print(f"\n{label}: {len(unmatched)} finding(s) could not be matched to a labeled row "
                  f"(quote text differs slightly, e.g. because of merged sentences):")
            for q in unmatched:
                print("   -", q[:100])

    print("\nPrecision = share of the run's own (non-artifact) findings that a human labeled 'real'.")
    print("This is precision only. Recall (how many real problems were found at all) needs a")
    print("ground truth built independently, e.g. from BGG rules-forum threads or errata.")


if __name__ == "__main__":
    main()
