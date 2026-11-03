"""Build an Excel labeling sheet from all detection runs of one rulebook.

Findings from every model/prompt run are merged by the sentence they point to,
so each problem appears once, with the list of runs that found it. The person
labeling fills in the "verdict" column (dropdown). Those labels give the
precision of each run and are the starting point for the ground truth.

Usage:
    python make_labeling_sheet.py --game Carcassonne
    python make_labeling_sheet.py --game CATAN --model haiku --missing-only

--missing-only puts into the sheet only the findings that a user would see (not marked as an
extraction artifact by the model) and whose sentence has no verdict yet in the other labeling
sheets of the same rulebook. It was used to label Haiku on Catan and Wingspan, which had been
labeled for Sonnet only; score_precision.py reads all the sheets of a rulebook together.
"""
import argparse
import glob
import json
import os
import sys

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # so that the other scripts are found when this one is started with python3 -I
from score_precision import load_verdicts  # noqa: E402

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labeling")

VERDICTS = ["real", "not_real", "extraction_artifact", "unsure"]
SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}

COLUMNS = [
    ("id", 6), ("page", 6), ("section", 18), ("sentence", 55), ("context (before / after)", 55),
    ("found by", 24), ("category (per run)", 26), ("model explanation", 60),
    ("suggested rewrite", 45), ("model says artifact", 12),
    ("YOUR VERDICT", 18), ("your notes", 35),
]


def run_label(path, data):
    model = "sonnet" if "sonnet" in data["model"] else "haiku"
    return f"{model}{'+hints' if data['hints'] else ''}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", required=True)
    parser.add_argument("--model", choices=["haiku", "sonnet"], default=None,
                        help="only include runs using this model (default: all runs)")
    parser.add_argument("--missing-only", action="store_true",
                        help="only findings a user would see that have no verdict yet in the other sheets of this rulebook")
    args = parser.parse_args()

    nlp_files = [f for f in glob.glob(os.path.join(OUTPUT_DIR, "*.nlp.json"))
                 if args.game.lower() in os.path.basename(f).lower()]
    if len(nlp_files) != 1:
        sys.exit(f"Expected one rulebook matching '{args.game}', found {nlp_files}")
    with open(nlp_files[0], encoding="utf-8") as f:
        nlp = json.load(f)
    sentences = {s["id"]: s for s in nlp["sentences"]}
    stem = os.path.basename(nlp_files[0]).replace(".nlp.json", "")
    suffix = f".{args.model}" if args.model else ""
    out = os.path.join(LABEL_DIR, f"{stem}{suffix}.labeling.xlsx")

    runs = sorted(glob.glob(os.path.join(OUTPUT_DIR, f"{stem}.*.detections.json")))
    if args.model:
        runs = [r for r in runs if f".{args.model}." in os.path.basename(r) or
                os.path.basename(r).split(".")[1] == args.model]
    if not runs:
        sys.exit("No detection runs found. Run detect.py first.")

    merged = {}  # primary sentence id -> merged finding
    for path in runs:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        label = run_label(path, data)
        for issue in data["issues"]:
            if not issue["sentence_ids"]:
                continue
            sid = issue["sentence_ids"][0]
            m = merged.setdefault(sid, {"runs": [], "cats": [], "explanations": [],
                                        "rewrite": "", "artifact": []})
            if label not in m["runs"]:
                m["runs"].append(label)
            m["cats"].append(f"{label}: {issue['category']} ({issue['severity']})")
            m["explanations"].append(f"[{label}] {issue['explanation']}")
            m["rewrite"] = m["rewrite"] or issue["suggested_rewrite"]
            m["artifact"].append(bool(issue.get("is_extraction_artifact")))

    if args.missing_only:
        done = {}  # the verdicts that exist already, by the exact text of the sentence
        for other in glob.glob(os.path.join(LABEL_DIR, f"{stem}*.labeling.xlsx")):
            if os.path.basename(other).startswith("~$") or os.path.abspath(other) == os.path.abspath(out):
                continue
            done.update(load_verdicts(other)[0])
        merged = {sid: m for sid, m in merged.items()
                  if sid in sentences and not all(m["artifact"]) and sentences[sid]["text"].strip() not in done}

    wb = Workbook()
    ws = wb.active
    ws.title = "Findings"
    ws.append([c for c, _ in COLUMNS])
    for i, (_, width) in enumerate(COLUMNS, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = width
        cell = ws.cell(row=1, column=i)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E79" if "YOUR" not in cell.value else "C55A11")
        cell.alignment = Alignment(wrap_text=True, vertical="center")

    for n, sid in enumerate(sorted(merged), start=1):
        m, s = merged[sid], sentences.get(sid)
        if s is None:
            continue
        before = sentences.get(sid - 1, {}).get("text", "")
        after = sentences.get(sid + 1, {}).get("text", "")
        artifact = ("yes" if all(m["artifact"]) else "no" if not any(m["artifact"]) else "mixed")
        ws.append([n, s["page"], s["section"], s["text"], f"{before}\n---\n{after}",
                   ", ".join(m["runs"]), "\n".join(m["cats"]), "\n\n".join(m["explanations"]),
                   m["rewrite"], artifact, "", ""])

    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    last = ws.max_row
    verdict_col = [c for c, _ in COLUMNS].index("YOUR VERDICT") + 1
    letter = ws.cell(row=1, column=verdict_col).column_letter
    dv = DataValidation(type="list", formula1='"' + ",".join(VERDICTS) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{letter}2:{letter}{max(last, 2)}")
    for row in range(2, last + 1):
        ws.cell(row=row, column=verdict_col).fill = PatternFill("solid", fgColor="FFF2CC")
    ws.freeze_panes = "E2"
    ws.auto_filter.ref = ws.dimensions

    guide = wb.create_sheet("How to label")
    for line in [
        "For each row, read the SENTENCE (and its context) and decide whether the RULEBOOK itself has a problem.",
        "Choose a verdict in the yellow YOUR VERDICT column:",
        "  real - the rulebook wording has a genuine problem (unclear, inconsistent, contradictory or unclear timing).",
        "  not_real - the sentence is fine, the model misread it or was too picky.",
        "  extraction_artifact - the text is scrambled because of the PDF (columns, callout boxes, artwork), not the rulebook.",
        "  unsure - you cannot decide. Add a note.",
        "Judge the sentence yourself first; open the PDF at the given page if the context is unclear.",
        "Do not use the model's explanation as proof. It can be wrong.",
        "If you spot a real problem the models missed, add a new row at the bottom with verdict 'real' and note 'missed'.",
    ]:
        guide.append([line])
    guide.column_dimensions["A"].width = 120

    os.makedirs(LABEL_DIR, exist_ok=True)
    wb.save(out)
    print(f"{len(merged)} unique findings{' without a verdict yet' if args.missing_only else ''} from {len(runs)} runs -> {out}")
    print("Runs:", ", ".join(run_label(p, json.load(open(p, encoding='utf-8'))) for p in runs))


if __name__ == "__main__":
    main()
