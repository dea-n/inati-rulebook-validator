"""Makes one two-page PDF of the invented card game for each language (needs Google Chrome), from language_samples.json. The PDFs are kept next to it, so the checks do not need Chrome. Usage: python3 make_fixture_pdfs.py"""
import json, os, subprocess, html
SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
samples = json.load(open(f"{SRC}/fixtures/language_samples.json", encoding="utf-8"))
CSS = "body{font:13pt/1.5 Georgia,'Times New Roman',serif;margin:0} .page{padding:30mm 25mm;page-break-after:always;height:230mm} h1{font-size:20pt} h2{font-size:14pt;margin:18px 0 6px;letter-spacing:.04em} p{margin:0 0 10px} @page{size:A4;margin:0}"
for language, sections in samples.items():
    if language.startswith("_"): continue
    def block(h, t): return f"<h2>{html.escape(h)}</h2><p>{html.escape(t)}</p>"
    page1 = "<h1>KARTENJAGD</h1>" + block(*sections[0]) + block(*sections[1])
    page2 = block(*sections[2])
    doc = f"<!doctype html><html lang='{language[:2].lower()}'><head><meta charset='utf-8'><style>{CSS}</style></head><body><div class='page'>{page1}</div><div class='page'>{page2}</div></body></html>"
    h = f"{OUT}/Kartenjagd_{language}.html"; open(h, "w", encoding="utf-8").write(doc)
    pdf = f"{OUT}/Kartenjagd_{language}.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", "file://" + h], capture_output=True, text=True)
    os.remove(h)
    print(language.ljust(11), os.path.getsize(pdf) // 1024, "KB")
