# Inati: AI-assisted validation of board game rulebooks

Inati reads a board game rulebook (a PDF) and lists the places where the text is hard to follow: unclear wording, inconsistent terms, contradictions and unclear timing. Each finding shows the quote, the page, an explanation, a suggested rewrite and a short FAQ entry. A person still has to decide whether a finding is a real problem.

This is the software part of the MSc dissertation *AI-Assisted Validation of Board Game Rulebooks Using Natural Language Processing* (MSc Artificial Intelligence and Data Science, University of York Europe Campus, 2026). It is a research prototype, not a product.

## How it works

PDF → text and pages (`parser.py`, with OCR as a fallback) → sentences and rule-based checks (`nlp_pipeline.py`) → Claude looks for issues chunk by chunk (`detect.py`) and for contradictions across the whole document (`detect_contradictions.py`) → FAQ entries (`generate_faq.py`) → FastAPI server (`api.py`) → React interface (`frontend/`).

The default model is Claude Haiku 4.5; Claude Sonnet 5 can be chosen on the upload page. A rulebook can be in English, German, French, Spanish, Italian, Portuguese, Albanian, Greek or Dutch, and the report can be written in any of these languages.

## What is in this repository

| Path | Content |
|---|---|
| `src/` | Python code: parsing, NLP, detection, FAQ stage, API |
| `src/fixtures/` | Test rulebooks: an invented card game in nine languages |
| `src/output/analysis/` | Saved analyses of four example rulebooks, so that the interface shows results without an API key |
| `frontend/` | The web interface (React and Vite) |
| `src/language_checks.py`, `src/scenario_checks.py` | Scripted checks that do not call the paid API |
| `src/make_labeling_sheet.py`, `src/score_precision.py`, `src/evaluation_summary.py` | Tools used for the evaluation; they need the labelling sheets, which are not included |

## Not included, and why

- The publisher PDFs of the four example rulebooks (Catan, Carcassonne, Terraforming Mars, Wingspan), because of copyright. For these four examples the links to the original pages and the *Ask a question* box therefore do not work. Both work for any PDF that you upload yourself.
- The extracted rulebook text, the labelling sheets and the dissertation text.
- The box art in `frontend/public/images/` belongs to the publishers and is used only to identify the four example games.

## Run it

Tested with Python 3.13 and Node.js 25 on macOS.

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd src
python -m uvicorn api:app --port 8000
```

Frontend (in a second terminal):

```bash
cd frontend
npm install
npm run dev -- --port 5173
```

Open http://localhost:5173. The four saved analyses open without an API key. To analyse a new PDF, set your own Anthropic API key in the terminal where the backend is started:

```bash
export ANTHROPIC_API_KEY="your-key"
```

An analysis of a rulebook of 6 to 16 pages costs roughly USD 0.03 to 0.11 with Haiku 4.5, and about twice as much with Sonnet 5. The first run also downloads the NLTK word list.

## Checks

```bash
cd src
python language_checks.py   # 25 checks of the multilingual support
python scenario_checks.py   # 23 checks of the parser, the API safeguards and the HTTP operations
```

Neither script calls the paid API (a stub replaces the client). All 48 checks pass on the author's machine, where the rulebook PDFs are in `src/datasets/`. Without those PDFs three checks fail (P5, A5 and Q5), because they open a rulebook from the dataset.

## Limits

- Prototype only: there is no login, no rate limit and no upload size limit on the server, and CORS is open. Run it on your own computer; every analysis spends the credit of the API key in use.
- In the evaluation described in the dissertation, about one in five findings was a real problem, according to the labelling described there. Every finding therefore needs a human decision. Precision and recall were measured on English rulebooks only.
- The other languages were tried only on the invented test rulebooks in `src/fixtures/`.


## Licence

No licence is granted. The repository is shared for the assessment of the dissertation.
