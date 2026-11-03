import json
import os
import re

import numpy as np
import pdfplumber
import pypdfium2 as pdfium

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Locate the folders next to this file, regardless of the working directory
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CHUNK_SIZE = 800  # maximum words per chunk
PREVIEW_CHARS = 300
OCR_MIN_CHARS = 20  # a page with fewer characters than this (and an image) is treated as scanned
OCR_RENDER_SCALE = 2.0  # ~144 DPI; higher improves OCR accuracy at the cost of speed

_ocr_engine = None  # loaded lazily - most rulebooks never need it, and loading costs ~1s


def get_ocr_engine():
    """Load the OCR engine once and reuse it. Uses RapidOCR (ONNX-based) rather than
    the more common Tesseract, because Tesseract needs a system-level Homebrew
    install that failed on this machine's macOS version without a large, risky
    source-compile of 24+ libraries (see Chapter 5 for the full story). RapidOCR and
    its PDF-rendering counterpart pypdfium2 are both plain pip packages instead."""
    global _ocr_engine
    if _ocr_engine is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr_engine = RapidOCR()
    return _ocr_engine


def ocr_page_image(pdfium_page, scale=OCR_RENDER_SCALE):
    """Render one PDF page to an image and run OCR on it - the fallback for a
    scanned page, where pdfplumber's text layer is empty because there isn't one."""
    image = pdfium_page.render(scale=scale).to_pil()
    result, _elapsed = get_ocr_engine()(np.array(image))
    if not result:
        return ""
    return " ".join(region_text for _box, region_text, _confidence in result)


def clean_text(text):
    """Remove empty lines and unnecessary characters from the text."""
    if not text:
        return ""
    # Drop InDesign print-slug lines (e.g. "WS_GoalMat_r2.indd 1 8/2/18 6:50 PM"),
    # including the variant where every character is doubled ("iinndddd")
    text = "\n".join(
        line for line in text.split("\n")
        if not re.search(r'\.indd|\.iinndddd', line)
    )
    # Remove trademark symbols, which are layout noise for analysis
    text = re.sub(r'[®™]', '', text)
    # Join decorative letter-spaced titles, e.g. "W I N G S P A N" -> "WINGSPAN"
    text = re.sub(r'\b(?:[A-Z] ){2,}[A-Z]\b', lambda m: m.group(0).replace(" ", ""), text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    return text


def chunk_text(text, chunk_size=CHUNK_SIZE):
    """Split the text into smaller chunks for AI analysis."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def chunk_pages(pages, chunk_size=CHUNK_SIZE):
    """Group pages into chunks of at most chunk_size words.

    A chunk never starts or ends in the middle of a page unless a single page
    is longer than chunk_size. Each chunk records the pages it covers, so that
    every issue found later can be traced back to the rulebook page.
    """
    chunks = []
    current_words, current_pages = [], []

    def flush():
        if current_words:
            chunks.append({
                "pages": [current_pages[0], current_pages[-1]],
                "word_count": len(current_words),
                "text": " ".join(current_words),
            })
        current_words.clear()
        current_pages.clear()

    for page in pages:
        words = page["text"].split()
        if not words:
            continue
        if len(words) > chunk_size:
            # A very long page is split on its own
            flush()
            for i in range(0, len(words), chunk_size):
                part = words[i:i + chunk_size]
                chunks.append({
                    "pages": [page["page"], page["page"]],
                    "word_count": len(part),
                    "text": " ".join(part),
                })
            continue
        if current_words and len(current_words) + len(words) > chunk_size:
            flush()
        current_words.extend(words)
        current_pages.append(page["page"])
    flush()
    return chunks


def find_gutter(words, page_width, tolerance=0.01):
    """Find the x position of a vertical gap between two text columns.

    Scans the central 30-70% of the page for the widest strip that (almost) no
    word crosses. Returns None if the page looks like a single column.
    """
    if len(words) < 40:
        return None
    runs, start = [], None
    x = page_width * 0.30
    while x <= page_width * 0.70:
        crossing = sum(1 for w in words if w["x0"] < x < w["x1"])
        if crossing <= tolerance * len(words):
            if start is None:
                start = x
        elif start is not None:
            runs.append((start, x))
            start = None
        x += 1.0
    if start is not None:
        runs.append((start, x))
    runs = [r for r in runs if r[1] - r[0] >= 6]
    if not runs:
        return None
    widest = max(runs, key=lambda r: r[1] - r[0])
    return (widest[0] + widest[1]) / 2


def words_to_text(words, line_tolerance=3):
    """Rebuild text lines from a list of pdfplumber words."""
    lines = []
    for w in sorted(words, key=lambda w: (w["top"], w["x0"])):
        if lines and abs(lines[-1][0] - w["top"]) <= line_tolerance:
            lines[-1][1].append(w)
        else:
            lines.append([w["top"], [w]])
    return "\n".join(
        " ".join(w["text"] for w in sorted(line[1], key=lambda w: w["x0"]))
        for line in lines
    )


def merge_spaced_letters(words, line_tolerance=3, min_letters=4):
    """Merge headings written with spaced letters ("G A M E  O V E R V I E W") into words.

    Otherwise such a heading is treated as many one-letter words and is split
    across the two columns. A gap clearly wider than the letter spacing marks
    a word boundary.
    """
    def is_letter(w):
        return len(w["text"]) == 1 and w["text"].isalpha() and w["text"].isupper()

    merged, run = [], []

    def flush_run():
        if len(run) < min_letters:
            merged.extend(run)
        else:
            gaps = sorted(b["x0"] - a["x1"] for a, b in zip(run, run[1:]))
            median_gap = gaps[len(gaps) // 2]
            text = run[0]["text"]
            for a, b in zip(run, run[1:]):
                text += (" " if b["x0"] - a["x1"] > 2 * max(median_gap, 0.5) else "") + b["text"]
            merged.append({
                "text": text,
                "x0": run[0]["x0"], "x1": run[-1]["x1"],
                "top": run[0]["top"], "bottom": run[0]["bottom"],
            })
        run.clear()

    for w in sorted(words, key=lambda w: (round(w["top"] / line_tolerance), w["x0"])):
        if is_letter(w) and (not run or abs(run[-1]["top"] - w["top"]) <= line_tolerance):
            run.append(w)
        else:
            flush_run()
            if is_letter(w):
                run.append(w)
            else:
                merged.append(w)
    flush_run()
    return merged


def extract_page_text(page):
    """Extract a page's text, reading two-column layouts column by column.

    pdfplumber's default extraction reads across the whole page line by line,
    which interleaves the two columns of a typical rulebook page.
    """
    words = merge_spaced_letters(page.extract_words())
    gutter = find_gutter(words, float(page.width))
    if gutter is None:
        return page.extract_text(), 1
    left = [w for w in words if (w["x0"] + w["x1"]) / 2 < gutter]
    right = [w for w in words if (w["x0"] + w["x1"]) / 2 >= gutter]
    return words_to_text(left) + "\n\n" + words_to_text(right), 2


def extract_pages(path):
    """Read a PDF and return a list of {page, text, columns, ocr} with cleaned text.

    Falls back to OCR for a page whose text layer is empty or near-empty but which
    does contain an image - i.e. a scanned page rather than a born-digital one.
    OCR text is noisier (no column detection, occasional misreads - see Chapter
    6.4), so such pages are marked "ocr": True for transparency rather than being
    silently treated the same as normal extraction.
    """
    pages = []
    pdfium_doc = None  # opened lazily - only scanned PDFs need it
    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            extracted, columns = extract_page_text(page)
            cleaned = clean_text(extracted)
            used_ocr = False

            if len(cleaned) < OCR_MIN_CHARS and page.images:
                if pdfium_doc is None:
                    pdfium_doc = pdfium.PdfDocument(path)
                ocr_text = clean_text(ocr_page_image(pdfium_doc[page_num - 1]))
                if len(ocr_text) > len(cleaned):
                    cleaned, used_ocr = ocr_text, True

            pages.append({
                "page": page_num,
                "text": cleaned,
                "columns": columns,
                "ocr": used_ocr,
            })
    return pages


def process_rulebooks():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: folder {DATASET_DIR} not found!")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary = []

    for filename in sorted(os.listdir(DATASET_DIR)):
        if not filename.endswith(".pdf"):
            continue

        path = os.path.join(DATASET_DIR, filename)
        print(f"Reading and cleaning: {filename}...")

        pages = extract_pages(path)
        chunks = chunk_pages(pages)
        full_text = "\n\n".join(p["text"] for p in pages if p["text"])
        total_words = len(full_text.split())
        empty_pages = [p["page"] for p in pages if not p["text"]]
        ocr_pages = [p["page"] for p in pages if p["ocr"]]

        stem = os.path.splitext(filename)[0]
        with open(os.path.join(OUTPUT_DIR, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "source": filename,
                "pages": len(pages),
                "words": total_words,
                "empty_pages": empty_pages,
                "ocr_pages": ocr_pages,
                "page_texts": pages,
                "chunks": chunks,
            }, f, ensure_ascii=False, indent=2)
        with open(os.path.join(OUTPUT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as f:
            f.write(full_text)

        two_col = sum(1 for p in pages if p["columns"] == 2)
        print(f"-> {len(pages)} pages ({two_col} two-column), {total_words} words, {len(chunks)} chunks")
        if ocr_pages:
            print(f"   OCR used on pages {ocr_pages} (no text layer found - scanned page)")
        if empty_pages:
            print(f"   WARNING: no text at all on pages {empty_pages} (scanned with no readable content, or a blank/cover page)")
        preview = " ".join(full_text[:PREVIEW_CHARS].split())
        print(f"   Preview: {preview}...\n")

        summary.append({
            "source": filename,
            "pages": len(pages),
            "words": total_words,
            "chunks": len(chunks),
            "empty_pages": len(empty_pages),
            "ocr_pages": len(ocr_pages),
        })

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"{'Rulebook':45} {'Pages':>5} {'Words':>6} {'Chunks':>6}")
    for s in summary:
        print(f"{s['source']:45} {s['pages']:>5} {s['words']:>6} {s['chunks']:>6}")
    print(f"\nSaved cleaned text and chunks to: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_rulebooks()
