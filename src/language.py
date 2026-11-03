"""Languages of the analysis: which languages are supported, how the language of a rulebook is detected, and the
note that tells the model in which language to read the rulebook and in which to write its answer.

The rulebook is analysed in its own language; it is never translated before the analysis. The report language only
decides in which language the explanations (and the FAQ) are written. English with an English report is the original
behaviour of the system, so language_note() is empty for it and the prompts stay exactly as they were evaluated.

Detection needs no extra library and no API call: it counts the most frequent function words of each language
(Greek is recognised by its alphabet). It is fast enough to run when a file is chosen, and the user can always
change the result.
"""
import re

LANGUAGES = ["English", "German", "French", "Spanish", "Italian", "Portuguese", "Albanian", "Greek", "Dutch"]
DEFAULT_LANGUAGE = "English"
CODES = {"en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
         "pt": "Portuguese", "sq": "Albanian", "el": "Greek", "nl": "Dutch"}

# The most frequent function words of each language (lower case). A few of them occur in several languages; the
# detector compares the share of tokens that belong to each list, so the words that only one language has decide.
_FUNCTION_WORDS = {
    "English": """the of and to a in is you that it for on are with as be this by or from each your can may must not at
        an if has have will all one other when then after before his her their they he she its into than only any no but
        so do does which who what also more most been was were there these those them we our us would could should
        about over under between during once both either while until""",
    "German": """der die das den dem des ein eine einen einem einer eines und oder aber nicht ist sind war wird werden
        wurde wenn als wie auch auf aus bei mit nach von vor zu zum zur zwischen durch für über unter um im am ins er
        sie es wir ihr ihre sein seine ihm ihn sich kann können muss müssen darf dürfen hat haben jeder jede jedes alle
        andere anderen dann danach noch nur sehr mehr bis keine kein diese dieser dieses diesen jetzt immer wieder so da
        hier dort ob dass""",
    "French": """le la les un une des du de d l et ou mais ne pas est sont était sera être a ont avoir au aux en dans sur
        sous avec sans pour par que qui quoi dont où ce cet cette ces se sa son ses leur leurs il elle ils elles on
        nous vous je tu y plus moins très tout tous toute toutes chaque autre autres lorsque quand alors puis ensuite si
        comme aussi peut peuvent doit doivent fait c n s j qu""",
    "Spanish": """el la los las un una unos unas y o pero no es son era será ser ha han haber de del al en con sin por
        para que quien cuyo donde este esta estos estas ese esa se su sus lo le les él ella ellos ellas usted ustedes
        yo tu más menos muy todo todos toda todas cada otro otra otros otras cuando entonces luego después antes si como
        también puede pueden debe deben hace hay mismo ya sobre entre hasta desde durante""",
    "Italian": """il lo la i gli le un uno una e o ma non è sono era sarà essere ha hanno avere di del dello della dei
        degli delle al allo alla ai agli alle in nel nello nella nei negli nelle con su sul sulla per tra fra che chi
        cui dove questo questa questi queste quello quella si suo sua suoi sue lui lei loro noi voi io tu più meno molto
        tutto tutti tutta tutte ogni altro altra altri altre quando allora poi dopo prima se come anche può possono deve
        devono fa già""",
    "Portuguese": """o a os as um uma uns umas e ou mas não é são era será ser há têm tem ter de do da dos das em no na
        nos nas com sem por para que quem cujo onde este esta estes estas esse essa se seu sua seus suas lhe lhes ele
        ela eles elas nós você vocês eu tu mais menos muito todo todos toda todas cada outro outra outros outras quando
        então depois antes se como também pode podem deve devem faz mesmo já sobre entre até desde durante ao aos à às
        pelo pela pelos pelas""",
    "Albanian": """dhe të e i në për me që një është janë ishte do nga si ka kanë nuk ose por edhe më së tij saj tyre ky
        kjo këtë këto ato ata ai ajo ne ju unë ti u ta mos pa vetëm çdo secili tjetër tjetri pas para kur nëse atëherë
        mund duhet duhen mbi nën midis deri tek po jo shumë gjithë këtu atje ndaj sipas gjatë pasi derisa""",
    "Greek": """ο η το οι τα των του της τον την τους τις και ή αλλά δεν είναι ήταν θα να σε από για με στο στη στον στην
        στα στους στις που ότι όταν αν κάθε άλλος άλλη άλλο ένας μία μια ένα αυτός αυτή αυτό αυτά πολύ μόνο μετά πριν""",
    "Dutch": """de het een en of maar niet is zijn was wordt worden werd als wanneer ook op aan in uit met van voor bij
        naar over onder tussen door om te tot tegen zonder hij zij wij jullie ik je u ze hun haar zich kan kunnen moet
        moeten mag mogen heeft hebben elke elk alle andere dan daarna nog alleen meer erg dit dat deze die geen dus hier
        daar""",
}
# Words of one letter are left out of the detection: scattered letters of artwork would otherwise count for the language
# that has the most one-letter words. They matter for the noise check of the NLP stage, which has to know that "a" or "y"
# is a real word, so they are listed here per language.
FUNCTION_WORDS = {language: frozenset(w for w in words.split() if len(w) >= 2) for language, words in _FUNCTION_WORDS.items()}
ONE_LETTER_WORDS = {
    "English": frozenset("ai"), "German": frozenset(), "French": frozenset(["a", "à", "y"]),
    "Spanish": frozenset("ayoeu"), "Italian": frozenset(["a", "e", "è", "o"]),
    "Portuguese": frozenset(["a", "e", "é", "o", "à"]), "Albanian": frozenset("eiu"),
    "Greek": frozenset(["ο", "η", "ή"]), "Dutch": frozenset("u"),
}

TOKEN = re.compile(r"[^\W\d_]+")           # a word: letters of any alphabet, apostrophes and digits split it
MIN_TOKENS = 12                            # fewer words than this give no reliable result
HIGH_SHARE, HIGH_MARGIN = 0.12, 0.08       # the share of function words of the best language, and its lead over the second
LOW_SHARE, LOW_MARGIN = 0.07, 0.03


def tokens(text):
    """The words of a text in lower case, without one-letter words."""
    return [w for w in TOKEN.findall(text.lower()) if len(w) >= 2]


def function_words(language):
    """The frequent function words of a language (an empty set for an unknown language)."""
    return FUNCTION_WORDS.get(language, frozenset())


def one_letter_words(language):
    """The words of one letter that are real words in a language ("a" and "i" in English)."""
    return ONE_LETTER_WORDS.get(language, frozenset())


def normalize_language(value, default=DEFAULT_LANGUAGE):
    """The canonical name of a language given as a name ("german") or a code ("de"). Empty and "auto" give the default;
    anything else raises ValueError, so that a client can never put free text into a prompt."""
    if value is None or str(value).strip().lower() in ("", "auto"):
        return default
    key = str(value).strip().lower()
    for name in LANGUAGES:
        if key == name.lower():
            return name
    if key in CODES:
        return CODES[key]
    raise ValueError(f"Unknown language '{value}'. Choose one of: {', '.join(LANGUAGES)}")


def detect_language(text):
    """Detect the language of a text.

    Returns {"language": name or None, "confidence": "high" | "low" | "none", "scores": {name: share}}. "none" means
    that the text is too short (or too mixed) to say; "low" means a likely language with a small lead, which the
    interface shows as uncertain.
    """
    words = tokens(text)
    letters = [c for c in text if c.isalpha()]
    if len(words) < MIN_TOKENS or len(letters) < 40:
        return {"language": None, "confidence": "none", "scores": {}}
    greek = sum(1 for c in letters if "Ͱ" <= c <= "Ͽ" or "ἀ" <= c <= "῿") / len(letters)
    scores = {name: round(sum(1 for w in words if w in vocab) / len(words), 3) for name, vocab in FUNCTION_WORDS.items()}
    if greek > 0.4:
        return {"language": "Greek", "confidence": "high", "scores": scores}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top >= HIGH_SHARE and top - second >= HIGH_MARGIN:
        confidence = "high"
    elif top >= LOW_SHARE and top - second >= LOW_MARGIN:
        confidence = "low"
    else:
        return {"language": None, "confidence": "none", "scores": scores}
    return {"language": best, "confidence": confidence, "scores": scores}


def detect_pdf_language(pdf_path, max_pages=8, min_pages=3, enough_words=800):
    """Detect the language of a PDF from the text of its first pages (the text layer only; no OCR, so a scanned
    file gives "none" and the user chooses the language). At least min_pages pages are read, because the first page of
    a rulebook is often a cover with lettering and artwork, and more pages are read until enough words are collected."""
    import pdfplumber
    text, pages_read = "", 0
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:max_pages]:
            text += " " + (page.extract_text() or "")
            pages_read += 1
            if pages_read >= min_pages and len(tokens(text)) >= enough_words:
                break
    result = detect_language(text)
    result["pages_read"] = pages_read
    return result


def language_note(doc_language, report_language, stage="analysis"):
    """The text that is added to the system prompt of a stage so that it knows the languages.

    stage "analysis" (the chunk pass and the whole-document pass) and stage "faq". It is empty when the rulebook and the
    report are both English: that is the original behaviour of the system, and the English prompts stay unchanged.
    """
    doc = doc_language or DEFAULT_LANGUAGE
    report = report_language or DEFAULT_LANGUAGE
    if doc == DEFAULT_LANGUAGE and report == DEFAULT_LANGUAGE:
        return ""
    if stage == "faq":
        return (
            "LANGUAGES\n"
            f"- The rulebook is written in {doc}. The quotes and the suggested rewrites that you receive are in {doc}; the\n"
            f"  explanations are in {report}.\n"
            f"- Write every \"question\" and every \"answer\" in {report}.\n"
            f"- Base each answer on the rewrite and the explanation. If you quote the rulebook, keep the quotation in {doc}.\n"
            "- Never refer to a page, a section or a footnote of the rulebook that is not in the information you received. If the\n"
            "  rulebook does not settle the question, say so plainly instead of pointing to a place where the answer might be."
        )
    lines = ["LANGUAGES", f"- The rulebook is written in {doc}. Read it and judge it in {doc}; do not translate it."]
    if doc != DEFAULT_LANGUAGE:
        lines.append(f"- The examples in the instructions above are English: apply every criterion to the equivalent wording in {doc}.")
    lines += [
        f"- \"quote\" is copied word for word from the rulebook, in {doc}.",
        f"- Write \"explanation\" in {report}.",
        f"- Write \"suggested_rewrite\" in {doc}, the language of the rulebook: it holds the corrected rulebook wording (for a",
        f"  contradiction, the corrected sentence or sentences). Put any comment on which wording to choose in \"explanation\", in {report}.",
        "- The values of \"category\" and \"severity\" stay the English words that are listed in the tool.",
    ]
    return "\n".join(lines)


def with_note(system_prompt, note):
    """The system prompt followed by the language note; the prompt itself when there is no note."""
    return f"{system_prompt}\n\n{note}" if note else system_prompt
