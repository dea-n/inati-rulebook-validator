"""Sprint 2 NLP pipeline: structure detection, sentence splitting and rule-based checks.

Reads the parsed pages saved by parser.py (output/<game>.json) and writes
output/<game>.nlp.json containing the sentences (with page and section) and a
list of candidate issues. The LLM stage (Sprint 3) consumes these sentences and
uses the rule-based candidates as extra evidence.
"""
import json
import os
import re
from collections import Counter, defaultdict

import nltk
import spacy

import language as language_support

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

LONG_SENTENCE_TOKENS = 45
NOISE_KNOWN_WORD_RATIO = 0.6  # below this share of known words, a sentence is treated as noise
SCATTERED_LETTER_RATIO = 0.15  # above this share of one-letter words, a sentence is treated as noise
REPEATED_WORD_RATIO = 0.2  # (other languages) above this share of words repeated next to each other: overlapping text

# --- Rule lists ---------------------------------------------------------------

VAGUE_TERMS = {
    "vague_quantifier": [
        "some", "several", "many", "few", "a few", "a number of", "various",
        "a lot of", "most", "numerous",
    ],
    "hedge": [
        "usually", "generally", "typically", "often", "sometimes", "normally",
        "probably", "possibly", "perhaps", "might", "could", "as needed",
        "if necessary", "if possible", "appropriate", "suitable", "sufficient",
        "and/or", "etc", "or so",
    ],
    "weak_obligation": ["should", "ideally", "preferably", "recommended"],
}

TIMING_CUES = [
    "simultaneously", "at the same time", "at any time", "immediately",
    "whenever", "at the end of", "at the start of", "at the beginning of",
    "before or after", "in any order", "as soon as",
]

REFERENCE_PRONOUNS = {"it", "they", "this", "that", "these", "those", "them"}

EXPLANATIONS = {
    "vague_quantifier": "'{m}' does not say how many. Players may disagree on the exact amount.",
    "hedge": "'{m}' is hedging language. It leaves it unclear whether the rule always applies.",
    "weak_obligation": "'{m}' is a recommendation, not a requirement. It is unclear whether players may ignore it.",
    "ambiguous_reference": "The sentence starts with '{m}', which refers back to something earlier. The reference may be unclear.",
    "timing_cue": "'{m}' describes when something happens. The exact order of events may be unclear.",
    "long_sentence": "The sentence has {m} words, which makes it hard to follow.",
}

BULLET_RE = re.compile(r"^\s*(?:\d+[.)]|[•●▪\-*])\s+")
END_PUNCT = (".", "!", "?", ":", ";", ",")


def load_known_words():
    """English word list from NLTK, used to detect garbled text from artwork."""
    try:
        from nltk.corpus import words
        return {w.lower() for w in words.words()}
    except LookupError:
        nltk.download("words", quiet=True)
        from nltk.corpus import words
        return {w.lower() for w in words.words()}


def is_garbled(sent, alpha, known_words, language="English"):
    """Detect text scrambled by artwork (cards, tokens, diagrams).

    A sentence is garbled if too few of its words are known English words (the
    lemma is checked too, because the NLTK list has no inflected forms), or if
    it is full of scattered single letters or numbers.

    For a rulebook in another language there is no word list, and a list built from the rulebook itself cannot tell
    clean from scrambled sentences, so the known-word test is left out. Scattered letters, numbers and words
    that are repeated next to each other (the mark of overlapping text) decide, and the model still marks what is
    left of the damage as an extraction artifact.
    """
    if language != "English":
        one_letter = language_support.one_letter_words(language)
        scattered = sum(
            1 for t in alpha if len(t) == 1 and t.lower_ not in one_letter and t.i != sent[0].i
        ) / len(alpha)
        numeric = sum(1 for t in sent if t.like_num or t.text.isdigit()) / len(sent)
        repeated = sum(1 for a, b in zip(alpha, alpha[1:]) if a.lower_ == b.lower_) / len(alpha)
        return scattered > SCATTERED_LETTER_RATIO or numeric > 0.5 or repeated > REPEATED_WORD_RATIO
    known = sum(
        1 for t in alpha
        if t.is_stop or t.lower_ in known_words or t.lemma_.lower() in known_words
    ) / len(alpha)
    # Ignore a list marker such as "C)" at the start of the sentence
    single_letters = sum(
        1 for t in alpha
        if len(t) == 1 and t.lower_ not in ("a", "i") and t.i != sent[0].i
    ) / len(alpha)
    numeric = sum(1 for t in sent if t.like_num or t.text.isdigit()) / len(sent)
    return known < NOISE_KNOWN_WORD_RATIO or single_letters > SCATTERED_LETTER_RATIO or numeric > 0.5


# --- Structure detection -------------------------------------------------------

def is_heading(line):
    """A short line without sentence punctuation that is upper case or Title Case."""
    stripped = line.strip()
    tokens = stripped.split()
    if not 1 <= len(tokens) <= 6 or stripped.endswith(END_PUNCT):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if len(letters) < 3 or len(letters) / len(stripped) < 0.6:
        return False
    return stripped.isupper() or stripped.istitle()


ASCII_LETTERS = "[A-Za-z]"
ANY_LETTERS = r"[^\W\d_]"  # letters of any alphabet, for the rulebooks that are not in English


def join_lines(lines, known_words, letters=ASCII_LETTERS):
    """Join wrapped lines. A word split by a hyphen at a line break is rejoined
    only if the joined word is a real word ("tempe-" + "rature"), so that
    genuine hyphens ("Earth-like") are kept."""
    text = ""
    for line in lines:
        if not text:
            text = line
        elif text.endswith("-") and line[:1].islower():
            prefix = re.search(rf"({letters}+)-$", text)
            suffix = re.match(rf"{letters}+", line)
            joined = (prefix.group(1) + suffix.group(0)).lower() if prefix and suffix else ""
            text = text[:-1] + line if joined in known_words else text + line
        else:
            text += " " + line
    return text


def split_blocks(page_text, known_words, letters=ASCII_LETTERS):
    """Turn a page into blocks of (kind, text): heading, rule (bullet/numbered) or paragraph."""
    blocks, paragraph = [], []

    def flush():
        if paragraph:
            blocks.append(("paragraph", join_lines(paragraph, known_words, letters)))
            paragraph.clear()

    for line in page_text.split("\n"):
        line = line.strip()
        if not line:
            flush()
        elif is_heading(line):
            flush()
            blocks.append(("heading", line))
        elif BULLET_RE.match(line):
            flush()
            blocks.append(("rule", BULLET_RE.sub("", line)))
        else:
            paragraph.append(line)
    flush()
    return blocks


# --- Checks --------------------------------------------------------------------

def find_term(text_lower, term):
    return re.search(r"(?<![\w/])" + re.escape(term) + r"(?![\w/])", text_lower)


def check_sentence(sent, text_lower):
    """Rule-based candidate issues for one sentence. Returns (category, rule, match) tuples."""
    found = []
    for category, terms in VAGUE_TERMS.items():
        for term in terms:
            m = find_term(text_lower, term)
            if m:
                found.append((category, term, m.group(0)))
    for cue in TIMING_CUES:
        m = find_term(text_lower, cue)
        if m:
            found.append(("timing_cue", cue, m.group(0)))
    first = sent[0]
    if first.lower_ in REFERENCE_PRONOUNS and first.pos_ == "PRON":
        found.append(("ambiguous_reference", first.lower_, first.text))
    if len(sent) > LONG_SENTENCE_TOKENS:
        found.append(("long_sentence", "length", str(len(sent))))
    return found


def term_key_and_form(chunk):
    """Normalised key and surface form of a noun chunk, ignoring plurals and case."""
    tokens = [t for t in chunk if t.pos_ not in ("DET", "PRON", "PUNCT") or t.text == "-"]
    tokens = [t for t in tokens if t.is_alpha or t.text == "-"]
    while tokens and tokens[0].text == "-":
        tokens.pop(0)
    while tokens and tokens[-1].text == "-":
        tokens.pop()
    if not 1 <= len(tokens) <= 3 or not any(t.is_alpha for t in tokens):
        return None, None
    form = ""
    for i, t in enumerate(tokens):
        piece = t.lemma_.lower() if t.is_alpha else t.text
        form += piece + (t.whitespace_ if i < len(tokens) - 1 else "")
    key = re.sub(r"[^a-z0-9]", "", form)
    return (key, form.strip()) if len(key) >= 6 else (None, None)


# --- Pipeline ------------------------------------------------------------------

def analyse(data, nlp, known_words, language="English"):
    """Sentences, structure and (English only) rule-based candidates of a parsed rulebook.

    language is the language of the rulebook. English is the original behaviour; for any other language the word list
    and the English rule checks are not used (see is_garbled), but the sentences and the sections are made in the same way.
    """
    sentences, issues = [], []
    term_forms = defaultdict(Counter)
    term_pages = defaultdict(set)
    section = "(start)"
    headings = []
    # Words that appear unbroken in this rulebook also count as real words
    # when deciding whether to rejoin a hyphenated line break
    english = language == "English"
    letters = ASCII_LETTERS if english else ANY_LETTERS
    doc_words = {
        w for p in data["page_texts"]
        for w in re.findall(r"[a-z]{4,}" if english else rf"{ANY_LETTERS}{{4,}}", p["text"].lower())
    }
    vocabulary = (known_words | doc_words) if english else doc_words

    for page in data["page_texts"]:
        for kind, text in split_blocks(page["text"], vocabulary, letters):
            if kind == "heading":
                section = text
                headings.append({"page": page["page"], "heading": text})
                continue
            for sent in nlp(text).sents:
                sent_text = sent.text.strip()
                alpha = [t for t in sent if t.is_alpha]
                if len(alpha) < 3:
                    continue
                is_noise = is_garbled(sent, alpha, known_words, language)
                sid = len(sentences)
                sentences.append({
                    "id": sid,
                    "page": page["page"],
                    "section": section,
                    "kind": kind,
                    "text": sent_text,
                    "tokens": len(sent),
                    "noise": is_noise,
                    "ocr": page.get("ocr", False),
                })
                if is_noise or not english:
                    continue  # the rule lists and the noun chunks below are English; the model does the analysis of other languages
                for category, rule, match in check_sentence(sent, sent_text.lower()):
                    issues.append({
                        "sentence_id": sid,
                        "page": page["page"],
                        "section": section,
                        "category": category,
                        "rule": rule,
                        "match": match,
                        "sentence": sent_text,
                        "explanation": EXPLANATIONS[category].format(m=match),
                    })
                for chunk in sent.noun_chunks:
                    key, form = term_key_and_form(chunk)
                    if key:
                        term_forms[key][form] += 1
                        term_pages[key].add(page["page"])

    terminology = []
    for key, forms in term_forms.items():
        if len(forms) >= 2 and sum(forms.values()) >= 3:
            terminology.append({
                "term": key,
                "variants": dict(forms.most_common()),
                "pages": sorted(term_pages[key]),
            })
            issues.append({
                "sentence_id": None,
                "page": min(term_pages[key]),
                "section": None,
                "category": "inconsistent_terminology",
                "rule": key,
                "match": " / ".join(forms),
                "sentence": None,
                "explanation": "The same term is written in different ways: "
                               + ", ".join(f"'{f}' ({n}x)" for f, n in forms.most_common()) + ".",
            })

    for n, issue in enumerate(issues):
        issue["id"] = n
    return {
        "source": data["source"],
        "stats": {
            "pages": data["pages"],
            "ocr_pages": len(data.get("ocr_pages", [])),
            "sentences": len(sentences),
            "noise_sentences": sum(1 for s in sentences if s["noise"]),
            "ocr_sentences": sum(1 for s in sentences if s["ocr"]),
            "avg_sentence_tokens": round(
                sum(s["tokens"] for s in sentences if not s["noise"])
                / max(1, sum(1 for s in sentences if not s["noise"])), 1),
            "headings": len(headings),
            "issues": dict(Counter(i["category"] for i in issues)),
        },
        "headings": headings,
        "sentences": sentences,
        "terminology": terminology,
        "issues": issues,
    }


def main():
    # output/ has accumulated many kinds of .json file since Sprint 3 (detections,
    # contradictions, faq, analysis...), not just parser.py's own output - checking
    # for "page_texts" (a key only parser.py's output has) is more robust than an
    # ever-growing list of suffixes to exclude by name.
    candidates = sorted(f for f in os.listdir(OUTPUT_DIR) if f.endswith(".json"))
    files = []
    for f in candidates:
        with open(os.path.join(OUTPUT_DIR, f), encoding="utf-8") as fh:
            if "page_texts" in json.load(fh):
                files.append(f)
    if not files:
        print("No parsed rulebooks found. Run parser.py first.")
        return

    nlp = spacy.load("en_core_web_sm")
    known_words = load_known_words()
    categories = sorted(set(EXPLANATIONS) | {"inconsistent_terminology"})
    results = []

    for filename in files:
        with open(os.path.join(OUTPUT_DIR, filename), encoding="utf-8") as f:
            data = json.load(f)
        print(f"Analysing: {data['source']}...")
        result = analyse(data, nlp, known_words)
        out_path = os.path.join(OUTPUT_DIR, filename.replace(".json", ".nlp.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        results.append(result)

    print(f"\n{'Rulebook':42} {'Sent.':>5} {'Noise':>5} {'AvgLen':>6} {'Head.':>5}")
    for r in results:
        s = r["stats"]
        print(f"{r['source'][:42]:42} {s['sentences']:>5} {s['noise_sentences']:>5} "
              f"{s['avg_sentence_tokens']:>6} {s['headings']:>5}")

    print("\nCandidate issues per category")
    print(f"{'Rulebook':42} " + " ".join(f"{c[:9]:>9}" for c in categories))
    for r in results:
        counts = r["stats"]["issues"]
        print(f"{r['source'][:42]:42} " + " ".join(f"{counts.get(c, 0):>9}" for c in categories))
    print(f"\nSaved to: {OUTPUT_DIR} (*.nlp.json)")


if __name__ == "__main__":
    main()
