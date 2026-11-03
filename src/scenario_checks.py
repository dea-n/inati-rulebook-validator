"""Scenario checks for the parser, the safeguards around the Claude API call and the HTTP operations.

Added at the end of the project (21 October 2026) to document the behaviour of the system and to make
the results of Chapter 5 reproducible. None of the checks calls the paid API: the API client is replaced
by a stub, and the HTTP operations are called through FastAPI's test client with the upload folder
redirected to a temporary directory.

Usage (from the src folder):
    python scenario_checks.py
    python scenario_checks.py --json results.json
"""
import argparse
import glob
import io
import json
import os
import random
import sys
import tempfile
import time
from types import SimpleNamespace

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import pdfplumber
import parser as rulebook_parser
import nlp_pipeline
import detect
import generate_faq
import api
from fastapi.testclient import TestClient

results = []


def check(group, cid, scenario, expected, fn):
    """Run one check. A crash counts as a failed check and its message is recorded."""
    t0 = time.time()
    try:
        ok, actual = fn()
    except Exception as e:
        ok, actual = False, f"exception: {type(e).__name__}: {str(e)[:120]}"
    results.append(dict(group=group, id=cid, scenario=scenario, expected=expected, actual=actual,
                        passed=bool(ok), seconds=round(time.time() - t0, 2)))
    print(f"{'PASS' if ok else 'FAIL'}  {cid:4} {scenario[:70]:70} -> {actual[:90]}")


# ------------------------------------------------------------------ parser and NLP stage
def word(x0, x1, top, text="w"):
    return {"x0": x0, "x1": x1, "top": top, "bottom": top + 10, "text": text}


def two_column_words():
    words = []
    for row in range(12):
        for k in range(5):
            words.append(word(50 + k * 45, 85 + k * 45, 40 + row * 14))      # left column, x 50 to 270
            words.append(word(330 + k * 45, 365 + k * 45, 40 + row * 14))    # right column, x 330 to 550
    return words


def single_column_words():
    rnd = random.Random(7)  # fixed seed; words of different widths, as in running text
    words = []
    for row in range(25):
        x = 100.0
        while x < 480:
            w = rnd.uniform(18, 60)
            words.append(word(x, x + w, 40 + row * 14))
            x += w + 4
    return words


def p1():
    g = rulebook_parser.find_gutter(two_column_words(), 600.0)
    ok = g is not None and 270 < g < 330
    return ok, f"gutter at x = {g:.0f} (columns end at 270 and start at 330)" if g else "None"


def p2():
    g1 = rulebook_parser.find_gutter(single_column_words(), 600.0)
    g2 = rulebook_parser.find_gutter(two_column_words()[:30], 600.0)
    return (g1 is None and g2 is None), \
        f"single column of running text (about {len(single_column_words())} words): {g1}; fewer than 40 words: {g2}"


def p2b():
    grid = [word(50 + (i % 10) * 50, 95 + (i % 10) * 50, 40 + (i // 10) * 14) for i in range(120)]  # aligned like a table
    g = rulebook_parser.find_gutter(grid, 600.0)
    return g is not None, (f"gutter reported at x = {g:.0f}: a table-like grid of aligned words is taken for a column gap"
                           if g else "None")


def p3():
    letters = [{"x0": 100 + i * 12, "x1": 108 + i * 12, "top": 50, "bottom": 60, "text": c} for i, c in enumerate("WINGSPAN")]
    out = rulebook_parser.merge_spaced_letters(letters)
    return (len(out) == 1 and out[0]["text"] == "WINGSPAN"), f"{len(out)} word(s): {[w['text'] for w in out]}"


def p4():
    raw = "WS_GoalMat_r2.indd 1 8/2/18 6:50 PM\nPlayers take turns.\nWINGSPAN® is a game™."
    out = rulebook_parser.clean_text(raw)
    ok = ".indd" not in out and "®" not in out and "™" not in out and "Players take turns." in out
    return ok, repr(out)


def p5():
    with pdfplumber.open(os.path.join(BASE_DIR, "datasets", "TerraformingMars_Rulebook.pdf")) as pdf:
        text, cols = rulebook_parser.extract_page_text(pdf.pages[4])
    first = text.splitlines()[1].strip()
    return (cols == 2 and first == "The game board has a map where tiles may be"), \
        f"columns = {cols}; first body line: '{first}'"


def p6():
    from PIL import Image, ImageDraw, ImageFont
    font_path = next(p for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
                                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf") if os.path.exists(p))
    font = ImageFont.truetype(font_path, 46)
    lines = ["Each player takes one turn in clockwise order.", "A player may build one road and one settlement.",
             "The game ends when a player reaches ten points."]
    img = Image.new("RGB", (1240, 400), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((40, 40 + i * 100), line, fill="black", font=font)
    path = os.path.join(tempfile.mkdtemp(), "synthetic_scan.pdf")
    img.save(path, "PDF", resolution=100.0)          # a PDF page that contains only an image
    pages = rulebook_parser.extract_pages(path)
    got = pages[0]["text"].lower()
    wanted = [w.strip(".").lower() for line in lines for w in line.split()]
    found = sum(1 for w in wanted if w in got)
    return (pages[0]["ocr"] is True and found / len(wanted) >= 0.9), \
        f"ocr = {pages[0]['ocr']}; {found} of {len(wanted)} words recovered ({found / len(wanted):.0%})"


def n1():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    known = nlp_pipeline.load_known_words()
    good, bad = nlp("Each player draws two cards at the start of the turn."), nlp("x 4 q Ww 12 7 zz k 3 9 b")
    noise_good = nlp_pipeline.is_garbled(good, [t for t in good if t.is_alpha], known)
    noise_bad = nlp_pipeline.is_garbled(bad, [t for t in bad if t.is_alpha], known)
    headings = [nlp_pipeline.is_heading(s) for s in ("GAME OVERVIEW", "Setting Up The Game", "Each player takes one turn.")]
    joined = nlp_pipeline.join_lines(["The tempe-", "rature rises"], known | {"temperature"})
    ok = noise_good is False and noise_bad is True and headings == [True, True, False] and "temperature" in joined
    return ok, (f"normal sentence noise={noise_good}; scrambled sentence noise={noise_bad}; headings={headings}; "
                f"joined='{joined}'")


check("parser", "P1", "find_gutter on a synthetic two-column page (120 words)", "a gutter between x = 270 and 330", p1)
check("parser", "P2", "find_gutter on a single-column page of running text and on a page with fewer than 40 words",
      "None in both cases", p2)
check("parser", "P2b", "find_gutter on a table-like grid of aligned words (known limitation)",
      "a gutter is reported, although the page has one block", p2b)
check("parser", "P3", "merge_spaced_letters on the letters W I N G S P A N", "one word, WINGSPAN", p3)
check("parser", "P4", "clean_text on a print-slug line and the symbols ® and ™", "slug and symbols removed, text kept", p4)
check("parser", "P5", "extract_page_text on page 5 of Terraforming Mars (two columns)",
      "two columns; first body line is the start of the left column", p5)
check("parser", "P6", "extract_pages on a synthetic scanned page (image only, three lines of text)",
      "page marked as OCR; at least 90% of the words recovered", p6)
check("nlp", "N1", "is_garbled, is_heading and the rejoining of a hyphenated word",
      "normal text kept, scrambled text marked, headings recognised, tempe-rature joined", n1)


# ------------------------------------------------------------------ safeguards around the API call (stub client, no cost)
def stub_client(tool_input, with_tool=True):
    block = SimpleNamespace(type="tool_use", input=tool_input) if with_tool else SimpleNamespace(type="text", text="hello")
    response = SimpleNamespace(content=[block], usage=SimpleNamespace(input_tokens=10, output_tokens=5))
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response
    return SimpleNamespace(messages=SimpleNamespace(create=create)), calls


ISSUE = dict(category="unclear_wording", sentence_ids=[1], quote="Some text.", explanation="x", severity="low",
             suggested_rewrite="y", is_extraction_artifact=False)


def s1():
    client, _ = stub_client({"issues": json.dumps([ISSUE])})
    issues, _usage = detect.call_claude(client, "m", "text")
    return len(issues) == 1, f"{len(issues)} finding decoded from a JSON string"


def s2():
    incomplete = dict(ISSUE)
    incomplete.pop("severity")
    client, _ = stub_client({"issues": [ISSUE, incomplete]})
    issues, _usage = detect.call_claude(client, "m", "text")
    return len(issues) == 1, f"{len(issues)} of 2 findings kept (the one without 'severity' dropped)"


def s3():
    client, _ = stub_client(None, with_tool=False)
    issues, _usage = detect.call_claude(client, "m", "text")
    return issues == [], f"result = {issues}"


def s4():
    chunk = [dict(id=1, text="The  player may draw a card,   then play it."), dict(id=2, text="Other sentence.")]
    base = dict(sentence_ids=[1], quote="The player may draw a card, then play it.")
    exact = detect.verify(base, chunk)[1]
    respaced = detect.verify(dict(base, quote="the PLAYER may draw a card, then play it."), chunk)[1]
    paraphrase = detect.verify(dict(base, quote="A player can take a card and play it."), chunk)[1]
    uncited = detect.verify(dict(base, quote="Other sentence."), chunk)[1]
    ids = detect.verify(dict(base, sentence_ids=[1, 99]), chunk)[0]
    ok = exact and respaced and not paraphrase and not uncited and ids == [1]
    return ok, (f"exact={exact}; other spacing and case={respaced}; paraphrase={paraphrase}; "
                f"quote from an uncited sentence={uncited}; unknown id dropped={ids == [1]}")


def s5():
    findings = [
        dict(id=0, category="unclear_wording", pages=[1], severity="low", quote="QUOTE-A", explanation="e", suggested_rewrite="r"),
        dict(id=1, category="unclear_timing", pages=[2], severity="low", quote="QUOTE-B", explanation="e", suggested_rewrite="r"),
        dict(id=2, category="contradiction", pages=[3], severity="high", quote="QUOTE-CONTRA", explanation="e", suggested_rewrite="r"),
    ]
    entries = [dict(issue_number=2, question="Q2", answer="A2", confidence="high")]  # the model skips entry 1
    client, calls = stub_client({"entries": entries})
    faq, review, _usage = generate_faq.generate_faq_entries(client, findings, quiet=True)
    sent = calls[0]["messages"][0]["content"]
    ok = (len(faq) == 1 and faq[0]["issue_id"] == 1 and faq[0]["question"] == "Q2"
          and len(review) == 1 and review[0]["issue_id"] == 2 and "QUOTE-CONTRA" not in sent)
    return ok, (f"answer for finding 2 attached to finding id {faq[0]['issue_id']} (not shifted to id 0); "
                f"1 skipped entry not guessed; contradiction in review list = {len(review) == 1}; "
                f"contradiction text sent to the model = {'QUOTE-CONTRA' in sent}")


check("api-call", "S1", "The list of findings arrives as a JSON string", "string decoded; one finding kept", s1)
check("api-call", "S2", "One of two findings lacks a required field", "incomplete entry discarded", s2)
check("api-call", "S3", "The response contains no tool call", "empty list, no crash", s3)
check("api-call", "S4", "verify on an exact quote, a re-spaced quote, a paraphrase and an uncited sentence",
      "True, True, False, False; unknown ids dropped", s4)
check("api-call", "S5", "FAQ stage: the model skips entry 1; one finding is a contradiction",
      "entry matched by number; contradiction not sent and put in the review list", s5)

# ------------------------------------------------------------------ HTTP operations (upload folder redirected to a temporary directory)
uploads = tempfile.mkdtemp()
api.UPLOAD_DIR = uploads
client = TestClient(api.app)


def a1():
    r = client.get("/")
    return r.status_code == 200 and r.json().get("status") == "ok", \
        f"HTTP {r.status_code}; {r.json().get('status')}; {len(r.json().get('endpoints', []))} endpoints listed"


def a2():
    r = client.get("/games")
    data = r.json()
    total = sum(g["issue_count"] for g in data)
    saved = glob.glob(os.path.join(api.ANALYSIS_DIR, "*.analysis.json"))   # the list has to match the analyses saved on disk
    on_disk = sum(len(json.load(open(f, encoding="utf-8"))["issues"]) for f in saved)
    return r.status_code == 200 and len(data) == len(saved) and total == on_disk, \
        f"HTTP {r.status_code}; {len(data)} rulebooks; {total} findings in total"


def a3():
    r = client.get("/games/Carcassonne_20th_Anniversary_Rulebook")
    d = r.json()
    ok = r.status_code == 200 and {"issues", "faq", "needs_human_review"} <= d.keys()
    return ok, f"HTTP {r.status_code}; {len(d['issues'])} findings, {len(d['faq'])} FAQ entry, {len(d['needs_human_review'])} entry for review"


def a4():
    r = client.get("/games/does_not_exist")
    return r.status_code == 404, f"HTTP {r.status_code}; {r.json().get('detail')}"


def a5():
    r = client.get("/pdf/CATAN–The Game Rulebook secure")
    disposition = r.headers.get("content-disposition", "")
    ok = (r.status_code == 200 and r.headers["content-type"] == "application/pdf" and disposition.startswith("inline")
          and "filename*=UTF-8''" in disposition and r.content[:4] == b"%PDF")
    return ok, f"HTTP {r.status_code}; {r.headers['content-type']}; Content-Disposition: {disposition}"


def a6():
    r = client.get("/pdf/does_not_exist")
    return r.status_code == 404, f"HTTP {r.status_code}"


def a7():
    r = client.post("/analyze", files={"file": ("notes.txt", b"hello", "text/plain")})
    return r.status_code == 400, f"HTTP {r.status_code}; {r.json().get('detail')}; files stored: {len(os.listdir(uploads))}"


def a8():
    r = client.post("/analyze", files={"file": ("broken.pdf", b"this is not a pdf", "application/pdf")})
    left = len(os.listdir(uploads))
    ok = r.status_code == 500 and "Analysis failed" in r.json().get("detail", "")
    return ok, f"HTTP {r.status_code}; {r.json().get('detail', '')[:60]}; the damaged upload stays on disk: {left} file(s)"


def a9():
    big = io.BytesIO(b"0" * (51 * 1024 * 1024))
    r = client.post("/analyze", files={"file": ("big.pdf", big, "application/pdf")})
    return r.status_code == 500, f"HTTP {r.status_code}: the server has no size limit; the file was rejected only when parsing failed"


def a10():
    r = client.post("/analyze")
    return r.status_code == 422, f"HTTP {r.status_code}"


check("http", "A1", "GET /", "200 and status ok", a1)
check("http", "A2", "GET /games", "200; one entry for each saved analysis", a2)
check("http", "A3", "GET /games/{id} for a saved analysis", "200 with findings, FAQ and review list", a3)
check("http", "A4", "GET /games/{id} for an unknown id", "404", a4)
check("http", "A5", "GET /pdf/{id} for the file name that contains an en dash",
      "200; application/pdf; inline; encoded file name", a5)
check("http", "A6", "GET /pdf/{id} for an unknown id", "404", a6)
check("http", "A7", "POST /analyze with a file that is not a PDF", "400 and nothing stored", a7)
check("http", "A8", "POST /analyze with a damaged file named .pdf", "500 with an error message (no API cost)", a8)
check("http", "A9", "POST /analyze with a 51 MB file named .pdf",
      "no size check on the server; fails when parsing (the interface limit is 50 MB)", a9)
check("http", "A10", "POST /analyze without a file", "422", a10)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--json", help="also write the results to this JSON file")
    args = ap.parse_args()
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed} of {len(results)} checks passed")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    sys.exit(0 if passed == len(results) else 1)
