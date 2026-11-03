import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'
import { saveReportPdf } from './exportReport.js'
import { labelSentences } from './sentenceRefs.js'

const API_BASE = 'http://localhost:8000'

const CATEGORY_LABELS = {
  unclear_wording: 'Unclear wording',
  inconsistent_terminology: 'Inconsistent terminology',
  contradiction: 'Contradiction',
  unclear_timing: 'Unclear timing',
}

// The model occasionally answers with a category that is not one of the four in its prompt
// (Terraforming Mars #15 came back as "unclear_terminology"). Map such near-misses onto the
// intended category, otherwise the badge shows the raw string with no colour and the finding
// is missing from the category breakdown.
const CATEGORY_ALIASES = { unclear_terminology: 'inconsistent_terminology' }

function normalizeResult(r) {
  const fix = (list) => (list || []).map((x) => ({ ...x, category: CATEGORY_ALIASES[x.category] || x.category }))
  return { ...r, issues: fix(r.issues), faq: fix(r.faq), needs_human_review: fix(r.needs_human_review) }
}

// The two models the server accepts (detect.MODELS in the backend). Haiku is the default: it is the
// model of the evaluation and of every saved analysis. The hints repeat what the evaluation found.
const MODEL_OPTIONS = [
  { id: 'haiku', label: 'Haiku 4.5', hint: 'Default · used in our evaluation · faster and cheaper' },
  { id: 'sonnet', label: 'Sonnet 5', hint: 'Optional · costs about 2 to 2.5 times as much as Haiku · was not more precise in our evaluation' },
]
const MODEL_LABELS = { 'claude-haiku-4-5-20251001': 'Haiku 4.5', 'claude-sonnet-5': 'Sonnet 5' }
const MODEL_KEYS = { 'claude-haiku-4-5-20251001': 'haiku', 'claude-sonnet-5': 'sonnet' }
const modelLabel = (modelId) => MODEL_LABELS[modelId] || modelId

// The languages the server accepts (language.LANGUAGES in the backend). A rulebook is analysed in its own language and is
// never translated; the report language only decides in which language the explanations and the FAQ are written.
const LANGUAGES = ['English', 'German', 'French', 'Spanish', 'Italian', 'Portuguese', 'Albanian', 'Greek', 'Dutch']
const DEFAULT_LANGUAGE = 'English'

// What the results header says about the languages of an analysis: nothing when everything is English, so an English
// rulebook looks exactly as before (and so do the analyses saved before languages existed, which have no language field).
function languageSummary(language) {
  if (!language || (language.document === DEFAULT_LANGUAGE && language.report === DEFAULT_LANGUAGE)) return null
  if (language.document === language.report) return language.document
  return `${language.document} rulebook · ${language.report} report`
}

// Each category gets its own accent colour, set via the data-category attribute
// below and styled in App.css, so a long results list is scannable by colour,
// not just by badge text.

// Box art for the four games in the dataset, sourced from each publisher's own
// site (the same publishers the rulebook PDFs came from). Matched by keyword so
// a freshly-uploaded new rulebook still falls back to a plain initial.
// import.meta.env.BASE_URL is Vite's configured `base` (see vite.config.js) - "/"
// in dev mode, "./" in the production build. Using it here (rather than a hardcoded
// leading "/") is what lets these images load correctly both ways, including when
// dist/index.html is opened directly by double-clicking it.
const GAME_IMAGES = [
  { match: /catan/i, src: `${import.meta.env.BASE_URL}images/catan.jpg` },
  { match: /carcassonne/i, src: `${import.meta.env.BASE_URL}images/carcassonne.jpg` },
  { match: /terraform/i, src: `${import.meta.env.BASE_URL}images/terraforming-mars.jpg` },
  { match: /wingspan|ws_/i, src: `${import.meta.env.BASE_URL}images/wingspan.jpg` },
]

function gameImage(sourceName) {
  return GAME_IMAGES.find((g) => g.match.test(sourceName))?.src || null
}

// Uploaded PDFs are stored as "<8 hex chars>_<original name>" (see api.analyze_pdf); show the original name.
const displayName = (source) => source.replace(/^[0-9a-f]{8}_/, '')

// The name of a rulebook as people say it. The example rulebooks have a fixed name. For any other file the name is
// cleaned: no upload prefix, extension, underscores or dashes, and no words that only describe the file ("Rulebook",
// "secure") or its version ("r24", "v2"). The original file name is shown small and grey where it helps.
const KNOWN_TITLES = [
  { match: /^catan\W+the game rulebook/i, name: 'Catan' },
  { match: /^carcassonne_20th_anniversary_rulebook/i, name: 'Carcassonne' },
  { match: /^terraformingmars_rulebook/i, name: 'Terraforming Mars' },
  { match: /^ws_rulebook/i, name: 'Wingspan' },
  { match: /^ws_appendix/i, name: 'Wingspan appendix' },
]
const FILE_WORDS = /^(rulebooks?|rules|secure|final|manual|v\d+(\.\d+)*|r\d{1,3})$/i

function cleanTitle(source) {
  const raw = displayName(source).replace(/\.pdf$/i, '')
  const known = KNOWN_TITLES.find((g) => g.match.test(raw))
  if (known) return known.name
  const words = raw
    .replace(/[_–—-]+/g, ' ')
    .replace(/([a-z])([A-Z])/g, '$1 $2')
    .split(/\s+/)
    .filter((w) => w && !FILE_WORDS.test(w))
  return words.join(' ') || raw
}

// What exportReport.js needs from this file to name things the way the interface does.
const reportLabels = {
  category: (c) => CATEGORY_LABELS[c] || c,
  model: modelLabel,
  title: (r) => cleanTitle(r.source),
}

function Logo({ className = '' }) {
  return (
    <img
      className={`brand-mark ${className}`}
      src={`${import.meta.env.BASE_URL}images/logo-white.png`}
      alt="Inati"
    />
  )
}

function SeverityBadge({ severity }) {
  if (!severity) return null
  return <span className={`badge badge-${severity}`}>{severity}</span>
}

function ConfidenceBadge({ confidence }) {
  return <span className={`badge badge-conf-${confidence}`}>{confidence} confidence</span>
}

// Makes a finding's page reference open the real PDF at that page (NFR3, Chapter
// 3.6): the backend serves the source file at /pdf/{gameId}, and the #page=N
// fragment is understood by the browser's built-in PDF viewer. Falls back to
// plain text if we don't know which rulebook this finding belongs to.
function PageLink({ pages, gameId }) {
  const label = `page${pages.length > 1 ? 's' : ''} ${pages.join(', ')}`
  if (!gameId || !pages.length) return <span className="pages">{label}</span>
  return (
    <a
      className="pages pages-link"
      href={`${API_BASE}/pdf/${encodeURIComponent(gameId)}#page=${pages[0]}`}
      target="_blank"
      rel="noreferrer"
      title="Open this page in the source PDF"
    >
      {label}
    </a>
  )
}

function IssueCard({ issue, gameId }) {
  return (
    <div className="card" data-category={issue.category}>
      <div className="card-header">
        <span className="badge badge-category">{CATEGORY_LABELS[issue.category] || issue.category}</span>
        <SeverityBadge severity={issue.severity} />
        <PageLink pages={issue.pages} gameId={gameId} />
      </div>
      <blockquote>&ldquo;{issue.quote}&rdquo;</blockquote>
      <p className="explanation">{issue.explanation}</p>
      {issue.suggested_rewrite && (
        <div className="rewrite">
          <strong>Suggested rewrite:</strong> {issue.suggested_rewrite}
        </div>
      )}
      {!issue.quote_verified && (
        <p className="warning">This quote could not be verified word-for-word against the source text.</p>
      )}
    </div>
  )
}

function FaqCard({ entry, gameId }) {
  return (
    <div className="card" data-category={entry.category}>
      <div className="card-header">
        <span className="badge badge-category">{CATEGORY_LABELS[entry.category] || entry.category}</span>
        <ConfidenceBadge confidence={entry.confidence} />
        <PageLink pages={entry.pages} gameId={gameId} />
      </div>
      <p className="question"><strong>Q:</strong> {entry.question}</p>
      <p className="answer"><strong>A:</strong> {entry.answer}</p>
    </div>
  )
}

// A contradiction candidate. The sentence numbers in its explanation are shown as letters (see sentenceRefs.js). Only
// Sentence A, the first sentence of the finding, is quoted, so the card says so and points to the page for the rest.
// The tag is "unverified" and not the severity: the model's severity says nothing when only 6% of these were real.
function ReviewCard({ item, gameId, sentenceIds }) {
  const labelled = sentenceIds.length > 0
  return (
    <div className="card card-review">
      <div className="card-header">
        <span className="badge badge-category">{CATEGORY_LABELS[item.category] || item.category}</span>
        <span
          className="badge badge-unverified"
          title="Not confirmed by a person. In the evaluation only 6% of the contradiction candidates were real."
        >
          unverified
        </span>
        <PageLink pages={item.pages} gameId={gameId} />
      </div>
      <blockquote>
        {labelled && (
          <span className="quote-label">
            Sentence A{item.pages.length === 1 ? ` · page ${item.pages[0]}` : ''}
          </span>
        )}
        &ldquo;{item.quote}&rdquo;
      </blockquote>
      {sentenceIds.length > 1 && (
        <p className="review-hint">Only Sentence A is quoted here. Open the page link to read the others in the rulebook.</p>
      )}
      <p className="explanation">{labelSentences(item.explanation, sentenceIds)}</p>
      <p className="review-reason"><strong>Needs human review:</strong> {item.reason}</p>
    </div>
  )
}

function CategoryBreakdown({ issues, activeFilter, onFilter }) {
  const counts = {}
  for (const i of issues) counts[i.category] = (counts[i.category] || 0) + 1
  const categories = Object.keys(CATEGORY_LABELS).filter((c) => c !== 'contradiction' && counts[c])
  const max = Math.max(1, ...categories.map((c) => counts[c]))

  // With a single category the bar would only repeat the total: there is nothing to compare or to filter.
  if (categories.length < 2) return null

  return (
    <div className="breakdown">
      {categories.map((c) => (
        <button
          key={c}
          className={`breakdown-row ${activeFilter === c ? 'active' : ''}`}
          onClick={() => onFilter(activeFilter === c ? null : c)}
        >
          <span className="breakdown-label">{CATEGORY_LABELS[c]}</span>
          <span className="breakdown-bar-track">
            <span
              className="breakdown-bar"
              data-category={c}
              style={{ width: `${(counts[c] / max) * 100}%` }}
            />
          </span>
          <span className="breakdown-count">{counts[c]}</span>
        </button>
      ))}
      {activeFilter && (
        <button className="breakdown-clear" onClick={() => onFilter(null)}>Clear filter</button>
      )}
    </div>
  )
}

// Saves the whole report (the issues, the FAQ and the contradiction candidates) as a PDF, whatever tab is open.
function SaveReport({ result }) {
  return (
    <div className="save-row">
      <button
        type="button"
        onClick={() => saveReportPdf(result, reportLabels)}
        title="Opens a printable page with the issues, the FAQ and the contradiction candidates. Choose Save as PDF in the print dialog."
      >
        Save as PDF
      </button>
    </div>
  )
}

// The pages and the quotes are shown whenever the server found sentences that the answer rests on, also when the rulebook
// only partly answers the question ("answerable" is false, and the answer says what the rulebook does say).
function AskAnswer({ item, gameId }) {
  const done = item.status === 'done'
  const hasSources = done && item.pages.length > 0
  const outcome = !done ? undefined : item.answerable ? 'yes' : hasSources ? 'partly' : 'no'
  const badge = { partly: 'Only partly answered', no: 'Not answered' }[outcome] || 'Your question'
  return (
    <div className="card ask-card" data-answered={outcome}>
      <div className="card-header">
        <span className="badge badge-category">{badge}</span>
        {hasSources && <PageLink pages={item.pages} gameId={gameId} />}
      </div>
      <p className="question"><strong>Q:</strong> {item.question}</p>
      {item.status === 'pending' && (
        <p className="ask-wait">Reading the rulebook<span className="chat-dots"><i /><i /><i /></span></p>
      )}
      {item.status === 'error' && <p className="warning">{item.error}</p>}
      {done && <p className="answer"><strong>A:</strong> {item.answer}</p>}
      {hasSources && item.quotes.length > 0 && (
        <details className="ask-quotes">
          <summary>The rulebook text the answer is based on</summary>
          {item.quotes.map((q, i) => <blockquote key={i}>&ldquo;{q}&rdquo;</blockquote>)}
        </details>
      )}
    </div>
  )
}

// A question about the rulebook, in any language. The server answers from the text of the rulebook only (POST /ask),
// gives the pages and the sentences the answer rests on, and says so when the rulebook does not answer the question.
// The questions and answers live in ResultsView, so they stay when the user switches to another tab and back.
function AskBox({ gameId, model, asked, setAsked }) {
  const [question, setQuestion] = useState('')
  const busy = asked.some((a) => a.status === 'pending')

  const update = (id, patch) => setAsked((list) => list.map((a) => (a.id === id ? { ...a, ...patch } : a)))

  const ask = (e) => {
    e.preventDefault()
    const text = question.trim()
    if (!text || busy) return
    const id = Date.now()
    setAsked((list) => [...list, { id, question: text, status: 'pending' }])
    setQuestion('')
    fetch(`${API_BASE}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId, question: text, model }),
    })
      .then(async (r) => {
        const body = await r.json().catch(() => ({}))
        if (!r.ok) throw new Error(typeof body.detail === 'string' ? body.detail : `Request failed (${r.status})`)
        return body
      })
      .then((answer) => update(id, { ...answer, status: 'done' }))
      .catch((err) =>
        update(id, {
          status: 'error',
          error: err instanceof TypeError ? `Could not reach the backend API at ${API_BASE}. Is it running?` : err.message,
        }),
      )
  }

  return (
    <section className="ask-box" aria-labelledby="ask-title">
      <h3 className="ask-title" id="ask-title">Ask a question about this rulebook</h3>
      <p className="ask-hint">
        Write it in any language and you get the answer in the same language. The answer comes from the text of the
        rulebook only, with the pages it is based on. If the rulebook does not say, you are told so.
      </p>
      <form className="ask-form" onSubmit={ask}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          maxLength={500}
          placeholder="e.g. when does my turn end?"
          aria-label="Your question"
          disabled={busy}
        />
        <button type="submit" disabled={busy || !question.trim()}>Ask</button>
      </form>
      {asked.length > 0 && (
        <div className="ask-thread" aria-live="polite">
          {asked.map((a) => <AskAnswer key={a.id} item={a} gameId={gameId} />)}
        </div>
      )}
    </section>
  )
}

function ResultsView({ result, onBack }) {
  const [tab, setTab] = useState('issues')
  const [categoryFilter, setCategoryFilter] = useState(null)
  const [asked, setAsked] = useState([])
  // All contradiction findings (local or whole-document) are routed to the
  // "Needs Review" tab by the backend (see generate_faq.generate_faq_entries),
  // not just document-scope ones - the Issues tab only shows the other 3 categories.
  const nonContra = result.issues.filter((i) => i.category !== 'contradiction')
  const visibleIssues = categoryFilter ? nonContra.filter((i) => i.category === categoryFilter) : nonContra
  // Matches the stem api.py uses for this game's saved files (see api.load_result),
  // so a finding's page link can find the right PDF at GET /pdf/{gameId}.
  const gameId = result.source.replace(/\.pdf$/i, '')
  const languages = languageSummary(result.language)
  // The sentences of each finding, so that a contradiction candidate can say which sentence it quotes.
  const issueById = new Map(result.issues.map((i) => [i.id, i]))

  return (
    <div className="results">
      <button className="back" onClick={onBack}>&larr; Back to rulebooks</button>
      <div className="results-head">
        <div className="results-title">
          <h2>{cleanTitle(result.source)}</h2>
          <p className="results-file">{displayName(result.source)}</p>
        </div>
        <SaveReport result={result} />
      </div>
      <div className="stats-row">
        <span>{result.stats.pages} pages</span>
        <span>{result.stats.sentences} sentences</span>
        <span>{nonContra.length} issue{nonContra.length !== 1 ? 's' : ''}</span>
        <span>{result.needs_human_review.length} needing review</span>
        {result.model && <span title="The model that analysed this rulebook">{modelLabel(result.model)}</span>}
        {languages && (
          <span className="stat-lang" title="The language of the rulebook, and the language of the explanations and the FAQ">
            {languages}
          </span>
        )}
        {result.stats.ocr_pages > 0 && (
          <span
            className="stat-ocr"
            title="These pages had no real text layer (likely scanned or a cover image), so text was read from the page image instead. OCR text is somewhat noisier than normally-extracted text."
          >
            {result.stats.ocr_pages} page{result.stats.ocr_pages !== 1 ? 's' : ''} via OCR
          </span>
        )}
      </div>

      <div className="tabs">
        <button className={tab === 'issues' ? 'active' : ''} onClick={() => setTab('issues')}>
          Issues ({nonContra.length})
        </button>
        <button className={tab === 'faq' ? 'active' : ''} onClick={() => setTab('faq')}>
          FAQ ({result.faq.length})
        </button>
        <button className={tab === 'review' ? 'active' : ''} onClick={() => setTab('review')}>
          Needs Review ({result.needs_human_review.length})
        </button>
      </div>

      {tab === 'issues' && (
        <>
          <CategoryBreakdown issues={nonContra} activeFilter={categoryFilter} onFilter={setCategoryFilter} />
          <div className="card-list">
            {visibleIssues.length === 0 && <p className="empty">No issues found in this category.</p>}
            {visibleIssues.map((issue) => (
              <IssueCard key={issue.id} issue={issue} gameId={gameId} />
            ))}
          </div>
        </>
      )}
      {tab === 'faq' && (
        <>
          <AskBox gameId={gameId} model={MODEL_KEYS[result.model] || MODEL_OPTIONS[0].id} asked={asked} setAsked={setAsked} />
          <div className="card-list">
            {result.faq.length === 0 && <p className="empty">No FAQ entries generated yet.</p>}
            {result.faq.length > 0 && (
              <p className="review-note">
                Drafted by the model from the rulebook text and not yet human-reviewed; the
                confidence label is the model&rsquo;s own estimate. Check important rulings
                against the source rulebook.
              </p>
            )}
            {result.faq.map((entry, i) => (
              <FaqCard key={i} entry={entry} gameId={gameId} />
            ))}
          </div>
        </>
      )}
      {tab === 'review' && (
        <div className="card-list">
          {result.needs_human_review.length === 0 && (
            <p className="empty">Nothing flagged for review - no contradiction candidates found.</p>
          )}
          <p className="review-note">
            Contradiction findings are routed here instead of an auto-generated FAQ answer,
            because our human-verified evaluation measured only 6% precision on this category
            (1 of 16 candidates confirmed genuine).
          </p>
          {result.needs_human_review.map((item, i) => (
            <ReviewCard key={i} item={item} gameId={gameId} sentenceIds={issueById.get(item.issue_id)?.sentence_ids || []} />
          ))}
        </div>
      )}
    </div>
  )
}

const svgProps = {
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 2,
  strokeLinecap: 'round',
  strokeLinejoin: 'round',
  'aria-hidden': true,
}

const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`

function GameCard({ game, onClick }) {
  const img = gameImage(game.source)
  const title = cleanTitle(game.source)
  return (
    <button type="button" className="game-card" onClick={onClick}>
      <div className="game-cover">
        {img ? <img src={img} alt={title} /> : <span className="game-initial">{title.charAt(0).toUpperCase()}</span>}
      </div>
      <div className="game-card-body">
        <strong>{title}</strong>
        <div className="game-meta">
          <span>
            <svg {...svgProps}>
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
            {plural(game.pages, 'page')}
          </span>
          <span>
            <svg {...svgProps}>
              <path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z" />
              <line x1="4" y1="22" x2="4" y2="15" />
            </svg>
            {plural(game.issue_count, 'finding')}
          </span>
        </div>
      </div>
    </button>
  )
}

const prefersReducedMotion = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches

// Scroll-snap carousel: the card nearest the middle is the active one (larger, glowing); the
// others are dimmed and clipped at the edges. Arrows, swipe/trackpad and clicking a side card
// all just move the scroll position - "active" is derived from it, never stored separately.
function GameCarousel({ games, loading, onSelect }) {
  const trackRef = useRef(null)
  const activeRef = useRef(0)
  const [active, setActive] = useState(0)

  const centerOn = useCallback((i, smooth) => {
    const track = trackRef.current
    const slide = track?.children[i]
    if (!slide) return
    track.scrollTo({
      left: slide.offsetLeft + slide.offsetWidth / 2 - track.clientWidth / 2,
      behavior: smooth && !prefersReducedMotion() ? 'smooth' : 'auto',
    })
  }, [])

  const syncActive = useCallback(() => {
    const track = trackRef.current
    if (!track) return
    const mid = track.scrollLeft + track.clientWidth / 2
    let best = 0
    let bestDist = Infinity
    for (let i = 0; i < track.children.length; i++) {
      const s = track.children[i]
      const d = Math.abs(s.offsetLeft + s.offsetWidth / 2 - mid)
      if (d < bestDist) {
        bestDist = d
        best = i
      }
    }
    activeRef.current = best
    setActive(best)
  }, [])

  // Start on the middle card so there is something on both sides of it.
  useEffect(() => {
    if (loading || games.length === 0) return
    centerOn(Math.floor((games.length - 1) / 2), false)
    syncActive()
  }, [loading, games.length, centerOn, syncActive])

  useEffect(() => {
    const onResize = () => centerOn(activeRef.current, false)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [centerOn])

  return (
    <div className="carousel" role="region" aria-roledescription="carousel" aria-label="Already analysed rulebooks">
      <div className="carousel-track" ref={trackRef} onScroll={syncActive}>
        {loading
          ? [0, 1, 2].map((i) => (
              <div className={`carousel-slide${i === 0 ? ' is-active' : ''}`} key={i}>
                <div className="game-card skeleton" />
              </div>
            ))
          : games.map((g, i) => (
              <div
                key={g.id}
                className={`carousel-slide${i === active ? ' is-active' : ''}`}
                role="group"
                aria-roledescription="slide"
                aria-label={`${i + 1} of ${games.length}`}
              >
                <GameCard game={g} onClick={() => (i === active ? onSelect(g.id) : centerOn(i, true))} />
              </div>
            ))}
      </div>
      {!loading && games.length > 1 && (
        <div className="carousel-nav">
          <button
            type="button"
            className="carousel-btn"
            onClick={() => centerOn(active - 1, true)}
            disabled={active === 0}
            aria-label="Previous rulebook"
          >
            <svg {...svgProps}><polyline points="15 18 9 12 15 6" /></svg>
          </button>
          <button
            type="button"
            className="carousel-btn"
            onClick={() => centerOn(active + 1, true)}
            disabled={active === games.length - 1}
            aria-label="Next rulebook"
          >
            <svg {...svgProps}><polyline points="9 18 15 12 9 6" /></svg>
          </button>
        </div>
      )}
      {!loading && games.length === 0 && (
        <p className="empty">No rulebooks analysed yet &mdash; press Upload Rulebook to upload the first one.</p>
      )}
    </div>
  )
}

// The UI enforces this limit itself (the backend only checks the .pdf extension).
const MAX_UPLOAD_BYTES = 50 * 1024 * 1024

function formatSize(bytes) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function LanguageSelect({ id, value, onChange, disabled }) {
  return (
    <select id={id} className="lang-select" value={value} onChange={(e) => onChange(e.target.value)} disabled={disabled}>
      {LANGUAGES.map((l) => <option key={l} value={l}>{l}</option>)}
    </select>
  )
}

// What the page says under the language of the chosen file. detection is null until a file is chosen, then
// { state: 'checking' | 'done' | 'failed' | 'manual', language, confidence } (see POST /detect-language).
function detectionHint(detection, language) {
  switch (detection?.state) {
    case 'checking':
      return { text: 'Detecting the language…' }
    case 'manual':
      return { text: 'Set by you.' }
    case 'failed':
      return { text: 'The language could not be detected. Choose the language of the rulebook.', warn: true }
    case 'done':
      if (!detection.language) {
        return {
          text: 'The language could not be detected (a scanned file, or very little text). English is assumed; change it if the rulebook is in another language.',
          warn: true,
        }
      }
      if (detection.confidence === 'low') return { text: `Probably ${language}, but not certain. Please check it.`, warn: true }
      return { text: 'Detected automatically. Change it if it is wrong.' }
    default:
      return { text: '' }
  }
}

function ChatUpload({ onUpload, uploading, uploadError }) {
  const [messages, setMessages] = useState([])
  const [pendingFile, setPendingFile] = useState(null)
  const [fileError, setFileError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [model, setModel] = useState(MODEL_OPTIONS[0].id)
  const [reportLanguage, setReportLanguage] = useState(DEFAULT_LANGUAGE)
  const [docLanguage, setDocLanguage] = useState(DEFAULT_LANGUAGE)
  const [detection, setDetection] = useState(null)
  const detectTicket = useRef(0)
  const wasUploading = useRef(false)
  const inputRef = useRef(null)
  const threadRef = useRef(null)

  // Before a file is sent the card is a drop zone; once sent it becomes the
  // conversation (the file as a message, then Inati's replies).
  const started = messages.length > 0
  const failed = messages[messages.length - 1]?.type === 'error'

  useEffect(() => {
    threadRef.current?.scrollTo({ top: threadRef.current.scrollHeight, behavior: 'smooth' })
  }, [messages])

  // The parent (App) owns the actual upload request. When it reports an error
  // after a request we started, surface it in the thread and drop the "typing"
  // placeholder - a success instead unmounts this whole view (see App: setResult
  // swaps ChatApp for ResultsView), so there's no separate success case here.
  useEffect(() => {
    if (wasUploading.current && !uploading && uploadError) {
      setMessages((m) => [
        ...m.filter((msg) => msg.type !== 'typing'),
        { role: 'ai', type: 'error', text: `I couldn't analyse that file: ${uploadError}` },
      ])
    }
    wasUploading.current = uploading
  }, [uploading, uploadError])

  // Asks the server for the language of the chosen file (POST /detect-language: it reads the first pages, costs
  // nothing and keeps nothing). The result only fills in the selector; the user can always change it.
  const detectLanguage = (f) => {
    const ticket = ++detectTicket.current // an answer for an earlier file, or one the user has overridden, is ignored
    setDocLanguage(DEFAULT_LANGUAGE)
    setDetection({ state: 'checking' })
    const form = new FormData()
    form.append('file', f)
    fetch(`${API_BASE}/detect-language`, { method: 'POST', body: form })
      .then((r) => {
        if (!r.ok) throw new Error(`Request failed (${r.status})`)
        return r.json()
      })
      .then((d) => {
        if (ticket !== detectTicket.current) return
        if (d.language) setDocLanguage(d.language)
        setDetection({ state: 'done', language: d.language, confidence: d.confidence })
      })
      .catch(() => {
        if (ticket === detectTicket.current) setDetection({ state: 'failed' })
      })
  }

  const acceptFile = (f) => {
    if (!f || uploading || started) return
    if (f.type !== 'application/pdf' && !/\.pdf$/i.test(f.name)) {
      setFileError('That file is not a PDF. Please choose a PDF rulebook.')
      return
    }
    if (f.size > MAX_UPLOAD_BYTES) {
      setFileError(`That file is ${formatSize(f.size)} — the maximum is 50 MB.`)
      return
    }
    setFileError(null)
    setPendingFile(f)
    detectLanguage(f)
  }

  const changeDocLanguage = (language) => {
    detectTicket.current += 1
    setDocLanguage(language)
    setDetection({ state: 'manual' })
  }

  const analyse = () => {
    if (!pendingFile || uploading || detection?.state === 'checking') return
    const file = pendingFile
    const chosen = MODEL_OPTIONS.find((o) => o.id === model)
    const languageNote = docLanguage === DEFAULT_LANGUAGE && reportLanguage === DEFAULT_LANGUAGE
      ? ''
      : ` Rulebook in ${docLanguage}, report in ${reportLanguage}.`
    setMessages([
      { role: 'user', type: 'file', text: file.name },
      {
        role: 'ai',
        type: 'typing',
        text: (model === 'haiku'
          ? `Analysing ${file.name} with ${chosen.label}… this usually takes under 2 minutes.`
          : `Analysing ${file.name} with ${chosen.label}… this may take longer than with Haiku.`) + languageNote,
      },
    ])
    setPendingFile(null)
    onUpload(file, model, { docLanguage, reportLanguage })
  }

  const reset = () => {
    setMessages([])
    setPendingFile(null)
    setFileError(null)
    setDocLanguage(DEFAULT_LANGUAGE)
    setDetection(null)
  }

  const openPicker = () => inputRef.current?.click()
  const errorLine = fileError && <p className="upload-error" role="alert">{fileError}</p>
  const hint = detectionHint(detection, docLanguage)

  return (
    <div className="upload-page">
      <header className="upload-head">
        <h1 className="upload-title">Analyze a Board Game Rulebook</h1>
        <p className="upload-sub">
          I&rsquo;ll check your PDF for unclear wording, contradictions, and more.<br />
          Upload your rulebook below to begin.
        </p>
      </header>

      {!started && (
        <div className="upload-options">
          <div className="model-picker">
            <span className="model-picker-label" id="model-picker-label">Model</span>
            <div className="tabs" role="radiogroup" aria-labelledby="model-picker-label">
              {MODEL_OPTIONS.map((o) => (
                <button
                  key={o.id}
                  type="button"
                  role="radio"
                  aria-checked={model === o.id}
                  className={model === o.id ? 'active' : ''}
                  onClick={() => setModel(o.id)}
                  disabled={uploading}
                >
                  {o.label}
                </button>
              ))}
            </div>
            <p className="model-picker-hint">{MODEL_OPTIONS.find((o) => o.id === model).hint}</p>
          </div>
          <div className="language-picker">
            <label className="language-picker-label" htmlFor="report-language">Report language</label>
            <LanguageSelect id="report-language" value={reportLanguage} onChange={setReportLanguage} disabled={uploading} />
            <p className="language-picker-hint">
              The explanations and the FAQ are written in this language. The rulebook is read in its own language and is never translated.
            </p>
          </div>
        </div>
      )}

      <div
        className={`upload-card${dragOver ? ' drag-over' : ''}`}
        onDragOver={(e) => { e.preventDefault(); if (!started && !uploading) setDragOver(true) }}
        onDragLeave={(e) => { if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(false) }}
        onDrop={(e) => {
          e.preventDefault()
          setDragOver(false)
          acceptFile(e.dataTransfer.files[0])
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          hidden
          onChange={(e) => { acceptFile(e.target.files[0]); e.target.value = '' }}
        />

        {started ? (
          <div className="upload-body">
            <div className="chat-thread" ref={threadRef}>
              {messages.map((m, i) => (
                <div key={i} className={`chat-msg chat-msg-${m.role}`}>
                  {m.role === 'ai' && <span className="chat-avatar">i</span>}
                  <div className={`chat-bubble ${m.type === 'file' ? 'chat-file' : ''}`}>
                    {m.type === 'file' && <span className="chat-file-icon">PDF</span>}
                    <span>{m.text}</span>
                    {m.type === 'typing' && <span className="chat-dots"><i /><i /><i /></span>}
                  </div>
                </div>
              ))}
            </div>
            {failed && (
              <div className="upload-retry">
                <button type="button" className="upload-select" onClick={reset}>Choose another file</button>
              </div>
            )}
          </div>
        ) : pendingFile ? (
          <div className="upload-zone upload-zone-file">
            <span className="upload-file-badge">PDF</span>
            <p className="upload-file-name">{pendingFile.name}</p>
            <p className="upload-file-size">{formatSize(pendingFile.size)}</p>
            <div className="upload-language">
              <label className="upload-language-label" htmlFor="doc-language">Rulebook language</label>
              <LanguageSelect id="doc-language" value={docLanguage} onChange={changeDocLanguage} disabled={uploading} />
              <p className={`upload-language-hint${hint.warn ? ' is-warn' : ''}`} role="status">{hint.text}</p>
            </div>
            <div className="upload-actions">
              <button type="button" className="upload-analyse" onClick={analyse} disabled={uploading || detection?.state === 'checking'}>
                Analyse rulebook
              </button>
              <button type="button" className="upload-select" onClick={openPicker} disabled={uploading}>
                Choose another file
              </button>
            </div>
            {errorLine}
          </div>
        ) : (
          <div className="upload-zone">
            <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="16 16 12 12 8 16" />
              <line x1="12" y1="12" x2="12" y2="21" />
              <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3" />
            </svg>
            <p className="upload-zone-title">Drag &amp; Drop Rulebook PDF here</p>
            <button type="button" className="upload-select" onClick={openPicker} disabled={uploading}>
              or Select File from Computer
            </button>
            {errorLine}
          </div>
        )}

        <div className="upload-strip">
          <div
            className={`upload-progress${uploading ? ' is-busy' : ''}`}
            role={uploading ? 'progressbar' : undefined}
            aria-label={uploading ? 'Analysing rulebook' : undefined}
            aria-hidden={uploading ? undefined : true}
          >
            <span />
          </div>
          <p className="upload-meta">
            Max. 50MB &bull; Accepted format: <b>PDF</b>
          </p>
        </div>
      </div>
    </div>
  )
}

function Footer() {
  const stack = ['Python', 'pdfplumber', 'spaCy', 'NLTK', 'Claude API', 'FastAPI', 'React']
  return (
    <footer className="footer">
      <div className="footer-bar">
        <button
          type="button"
          className="footer-brand"
          onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
        >
          <Logo className="footer-logo" />
        </button>
        <span className="footer-copyright">&copy; 2026 Inati &middot; MSc AI &amp; Data Science capstone project</span>
        <div className="footer-tech">
          {stack.map((t) => <span key={t} className="tech-pill">{t}</span>)}
        </div>
      </div>
    </footer>
  )
}

// The game id Terraforming Mars is stored under (GET /games), used to open its full report.
const SHOWCASE_ID = 'TerraformingMars_Rulebook'
const SHOWCASE_FINDINGS = 34   // all findings of that analysis (issues + needing review)
const SHOWCASE_SENTENCES = 559
const SHOWCASE_PAGES = 16

function LiveExample({ onSelect }) {
  // Real output of the deployed analysis of Terraforming Mars (issues #12, #15, #11), copied
  // verbatim apart from #15's category, which the model returned as "unclear_terminology"
  // (see normalizeCategory). The note under the card says "human-verified", and that holds
  // only because a human labeled each of these sentences "real" in
  // output/labeling/TerraformingMars_Rulebook.labeling.xlsx - only add findings with that verdict.
  const findings = [
    {
      id: 12,
      category: 'unclear_timing',
      severity: 'medium',
      pages: [8],
      quote: 'You can choose to take 1 or 2 actions on your turn.',
      explanation: "The rule states players can take '1 or 2 actions' but then says 'if you take no action at all (pass), you are out.' This creates ambiguity: can a player choose to take 0 actions as a valid turn action, or does 'passing' mean they must remain in the round but simply skip their turn for that round? The phrasing suggests 'choose to take 1 or 2' implies these are the only valid options, yet the next sentence permits 'no action at all.'",
      suggested_rewrite: 'On your turn, you can choose to take 1 or 2 actions, or you can pass (take no actions), which removes you from the round for the rest of the generation.',
      quote_verified: true,
    },
    {
      id: 15,
      category: 'inconsistent_terminology',
      severity: 'medium',
      pages: [9],
      quote: 'Resources or production with a red frame means that you can affect any player you choose (even yourself).',
      explanation: "The rule uses both 'red frame' (S301) and 'red border' (S304) to describe the same concept, creating inconsistent terminology. Additionally, S304-305 introduce 'red' tiles and tags separately, but S301 already uses 'red' to mean something different (any player's resource/production). This could confuse players about what 'red' means in different contexts.",
      suggested_rewrite: 'Resources or production with a red border means that you can affect any player you choose (even yourself).',
      quote_verified: true,
    },
    {
      id: 11,
      category: 'unclear_wording',
      severity: 'low',
      pages: [7],
      quote: 'You start with 1 production of each resource on the player board, so place player markers on the tracks (only in standard game.)',
      explanation: "The phrase 'You start with 1 production of each resource' is ambiguous - it could mean 1 production point in each of the 6 resource tracks (as shown in sentence 208), or it could mean something else. The parenthetical clarification helps but the main clause is still vague about which resources and how many tracks.",
      suggested_rewrite: "You start with 1 production of each resource, so place player markers on '1' for each of the six production tracks (only in standard game.)",
      quote_verified: true,
    },
  ]
  const inDocumentOrder = [...findings].sort((a, b) => a.pages[0] - b.pages[0])
  const legend = [...new Set(inDocumentOrder.map((f) => f.category))]

  return (
    <div className="showcase" id="example">
      <div className="showcase-chrome">
        <span className="showcase-dot" />
        <span className="showcase-dot" />
        <span className="showcase-dot" />
        <span className="showcase-path">/rulebooks/terraforming-mars.pdf</span>
      </div>
      <div className="showcase-grid">
        <div className="showcase-doc">
          <div className="showcase-doc-stat"><strong>{SHOWCASE_FINDINGS}</strong>findings</div>
          <p className="showcase-doc-heading">Rulebook excerpts flagged by Inati</p>
          {/* The highlight takes the colour of the category tag of the same finding on the right */}
          <ul className="showcase-legend" aria-label="Highlight colours">
            {legend.map((c) => (
              <li key={c} data-category={c}>{CATEGORY_LABELS[c]}</li>
            ))}
          </ul>
          {inDocumentOrder.map((f) => (
            <p className="showcase-doc-line" key={f.id}>
              <span className="showcase-doc-page">p.{f.pages[0]}</span>
              <mark className="showcase-highlight" data-category={f.category}>&ldquo;{f.quote}&rdquo;</mark>
            </p>
          ))}
          <p className="showcase-doc-fade">
            &hellip; {SHOWCASE_SENTENCES - findings.length} more sentences analysed across {SHOWCASE_PAGES} pages &hellip;
          </p>
        </div>
        <div className="showcase-analysis">
          <div className="showcase-analysis-head">
            <h3>AI Analysis</h3>
            <span className="showcase-analysis-count">{findings.length} of {SHOWCASE_FINDINGS} shown</span>
          </div>
          {/* The column is as tall as the excerpts next to it; the cards scroll inside it */}
          <div className="showcase-scroll">
            <div className="card-list">
              {findings.map((f) => (
                <IssueCard key={f.id} issue={f} gameId={SHOWCASE_ID} />
              ))}
            </div>
          </div>
        </div>
      </div>
      <div className="showcase-footer">
        <span className="showcase-footer-note">Real findings, human-verified &middot; Terraforming Mars</span>
        <button type="button" className="showcase-view-report" onClick={() => onSelect(SHOWCASE_ID)}>
          View full report &rarr;
        </button>
      </div>
    </div>
  )
}

const stepIcons = {
  read: (
    <>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </>
  ),
  organise: (
    <>
      <line x1="8" y1="6" x2="21" y2="6" />
      <line x1="8" y1="12" x2="21" y2="12" />
      <line x1="8" y1="18" x2="21" y2="18" />
      <line x1="3" y1="6" x2="3.01" y2="6" />
      <line x1="3" y1="12" x2="3.01" y2="12" />
      <line x1="3" y1="18" x2="3.01" y2="18" />
    </>
  ),
  check: (
    <>
      <circle cx="11" cy="11" r="8" />
      <line x1="21" y1="21" x2="16.65" y2="16.65" />
    </>
  ),
  explain: <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />,
}

// Plain words for a visitor who knows nothing about NLP; the tools behind each step are named in the footer.
function HowItWorks() {
  const steps = [
    { icon: 'read', title: 'Read', text: 'Inati reads your PDF and pulls out clean text, even from pages with several columns.' },
    { icon: 'organise', title: 'Organise', text: 'The text is split into sentences and sections. Anything that looks like a printing or scanning error is marked.' },
    { icon: 'check', title: 'Check', text: 'Each passage, and then the whole rulebook, is checked for unclear wording, mixed-up terms, unclear timing and contradictions.' },
    { icon: 'explain', title: 'Explain', text: 'Each finding gets a plain-language explanation and a suggested rewrite. Most also get a draft FAQ answer for players.' },
  ]
  return (
    <section className="how-it-works" id="how-it-works">
      <h2>How it works</h2>
      <div className="steps-grid">
        {steps.map((s, i) => (
          <div className="step" key={s.title}>
            <div className="step-top">
              <span className="step-icon"><svg {...svgProps}>{stepIcons[s.icon]}</svg></span>
              <span className="step-number" aria-label={`Step ${i + 1}`}>{i + 1}</span>
            </div>
            <strong>{s.title}</strong>
            <p>{s.text}</p>
          </div>
        ))}
      </div>
    </section>
  )
}

// Real, human-verified numbers from Chapter 6 (output/evaluation/recall_evaluation.md),
// weak ones included on purpose: this is a research project, and an unqualified
// "accuracy" claim would misrepresent it. lo/hi bound each meter - recall against a
// small ground-truth set is a range, not a point estimate.
// Both models were labelled on all four rulebooks (Haiku on Catan and Wingspan on 1 November 2026); the cards show
// the deployed model, and `compare` gives the precision of the other model.
const EVAL_DATA = [
  {
    game: 'Carcassonne', sub: 'Simple rulebook', model: 'Haiku', deployed: true,
    precision: { lo: 100, hi: 100, shown: '100%', caption: '4 of 4 findings confirmed real' },
    recall: { lo: 40, hi: 50, shown: '~40–50%', caption: '2 of 4–5 documented issues found' },
    compare: 'Sonnet, not deployed: 57% precision (4 of 7 findings confirmed real)',
  },
  {
    game: 'Terraforming Mars', sub: 'Complex rulebook', model: 'Haiku', deployed: true,
    precision: { lo: 26, hi: 26, shown: '26%', caption: '7 of 27 findings confirmed real' },
    recall: { lo: 60, hi: 75, shown: '~60–75%', caption: '3 of 4–5 documented issues found' },
    compare: 'Sonnet, not deployed: 0% precision (0 of 6 findings confirmed real)',
  },
  {
    game: 'Catan', sub: 'Rulebook', model: 'Haiku', deployed: true,
    precision: { lo: 0, hi: 0, shown: '0%', caption: '0 of 9 findings confirmed real' },
    recall: { lo: 0, hi: 0, shown: '0%', caption: '0 of 5 documented issues found' },
    compare: 'Sonnet, not deployed: 17% precision (1 of 6 findings confirmed real)',
  },
  {
    game: 'Wingspan', sub: 'Rulebook + appendix', model: 'Haiku', deployed: true,
    precision: { lo: 0, hi: 0, shown: '0%', caption: '0 of 17 findings confirmed real' },
    recall: { lo: 0, hi: 0, shown: '0%', caption: '0 of 3 testable documented issues found' },
    compare: 'Sonnet, not deployed: 0% precision (0 of 5 findings confirmed real)',
  },
]

function useInView() {
  const ref = useRef(null)
  const [inView, setInView] = useState(false)
  useEffect(() => {
    const el = ref.current
    if (!el || typeof IntersectionObserver === 'undefined') {
      setInView(true)
      return undefined
    }
    const io = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) {
        setInView(true)
        io.disconnect()
      }
    }, { threshold: 0.15 })
    io.observe(el)
    return () => io.disconnect()
  }, [])
  return [ref, inView]
}

function Meter({ label, lo, hi, shown, caption, delay }) {
  const exact = lo === hi
  return (
    <div className={`meter${hi === 0 ? ' is-zero' : ''}`} style={{ '--lo': lo, '--hi': hi, '--delay': `${delay}ms` }}>
      <div className="meter-head">
        <span className="meter-label">{label}</span>
        <span className="meter-value">{shown}</span>
      </div>
      <div
        className="meter-track"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={lo}
        aria-valuetext={exact ? `${lo} percent` : `about ${lo} to ${hi} percent`}
      >
        <span className="meter-fill" />
        {!exact && <span className="meter-range" />}
      </div>
      <p className="meter-caption">{caption}</p>
    </div>
  )
}

function EvaluationTrust() {
  const [ref, inView] = useInView()
  return (
    <section ref={ref} className={`eval-section${inView ? ' in-view' : ''}`} id="evaluation">
      <h2>Evaluated honestly, not just claimed</h2>
      <p className="eval-intro">
        Every number below comes from a human reading every flagged sentence and comparing
        results against real, independently-sourced player confusion (BoardGameGeek threads
        and official FAQs) &mdash; not self-reported accuracy.
      </p>
      {/* EVAL_DATA above is Haiku for all four rulebooks, and Haiku is the deployed model */}
      <p className="eval-model-note">
        <span className="eval-model-chip">Haiku 4.5 &middot; deployed</span>
        All results below use Haiku 4.5, the deployed model. The result of the other model, Sonnet 5, is under each card.
      </p>
      <div className="eval-key">
        <p><strong>Precision</strong>Of everything Inati flagged, how much a human confirmed was a real problem.</p>
        <p><strong>Recall</strong>Of the problems players documented online, how many Inati caught.</p>
      </div>
      <div className="eval-grid">
        {EVAL_DATA.map((d, i) => {
          const cover = gameImage(d.game)
          return (
            <article className="eval-card" key={d.game}>
              <div className="eval-card-head">
                {cover && <img className="eval-cover" src={cover} alt="" />}
                <div className="eval-card-title">
                  <h3>{d.game}</h3>
                  <span>{d.sub}</span>
                </div>
              </div>
              <Meter label="Precision" {...d.precision} delay={i * 120} />
              <Meter label="Recall" {...d.recall} delay={i * 120 + 160} />
              {d.compare && <p className="eval-compare">{d.compare}</p>}
            </article>
          )
        })}
      </div>
      <p className="eval-note">
        The finding counts on these cards come from the evaluation runs, which were labelled by hand. The totals under
        &ldquo;Already analysed&rdquo; come from the saved analyses of the same model. Those are separate runs, and they also
        include the contradiction candidates. The model can answer a little differently each time, so the counts differ.
      </p>
      <div className="eval-callout">
        <strong>Limitations, stated plainly</strong>
        <ul>
          <li>
            Accuracy depends heavily on the rulebook, from 100% on a short one to 0% on two
            widely played ones. On Catan and Wingspan none of the 26 findings of Haiku was
            confirmed as a real problem: the text of these two rulebooks is already clear in
            most places, and Inati reads only text, not pictures. Both models were checked on
            all four rulebooks; the result of the other model is under each card.
          </li>
          <li>
            These are published rulebooks, not drafts of independent designers, who are the
            intended users. That case was not tested.
          </li>
          <li>
            Cross-page contradiction detection specifically measured only 6% precision (1 of 16
            candidates confirmed genuine) &mdash; which is why contradiction findings are routed
            to human review instead of an automatic answer.
          </li>
          <li>
            Sample sizes are small (3&ndash;5 documented issues per game): every number is a
            direction, not a precise measurement.
          </li>
        </ul>
      </div>
    </section>
  )
}

const NAV_SECTIONS = ['how-it-works', 'games', 'evaluation']

// The section of the home page that is being read: the last one whose top has passed the upper third of the window.
function useActiveSection(enabled) {
  const [active, setActive] = useState(null)
  useEffect(() => {
    if (!enabled) return undefined
    const update = () => {
      let current = null
      for (const id of NAV_SECTIONS) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top <= window.innerHeight * 0.35) current = id
      }
      setActive(current)
    }
    const first = requestAnimationFrame(update) // once the sections are on the screen
    window.addEventListener('scroll', update, { passive: true })
    window.addEventListener('resize', update)
    return () => {
      cancelAnimationFrame(first)
      window.removeEventListener('scroll', update)
      window.removeEventListener('resize', update)
    }
  }, [enabled])
  return enabled ? active : null
}

// view is 'home', 'report' (a rulebook is open) or 'upload'. On the home page the link of the section that is being read
// is marked; on a report the current place is the list of rulebooks; on the upload page no link is current and the
// "Get started" button is left out, because the user is already there.
function NavBar({ view, onHome, onOpenUpload }) {
  const onHomePage = view === 'home'
  const scrolledTo = useActiveSection(onHomePage)
  const current = view === 'report' ? 'games' : scrolledTo

  // Nav links are anchors to sections on the home page. If we're currently viewing a rulebook's results, clicking one
  // first goes back home (so the section actually exists on screen), then scrolls to it.
  const goTo = (id) => (e) => {
    if (onHomePage) return // already home - let the <a href="#..."> handle it natively
    e.preventDefault()
    onHome()
    // wait one frame for HomeView to mount before scrolling to the section
    requestAnimationFrame(() => {
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' })
    })
  }
  const link = (id, label) => (
    <a
      href={`#${id}`}
      onClick={goTo(id)}
      className={current === id ? 'is-active' : undefined}
      aria-current={current === id ? (view === 'report' ? 'page' : 'location') : undefined}
    >
      {label}
    </a>
  )

  return (
    <nav className="navbar">
      <button className="navbar-brand" onClick={onHome}><Logo /></button>
      <div className="navbar-links">
        {link('how-it-works', 'Overview')}
        {link('games', 'Rulebook')}
        {link('evaluation', 'Evaluation')}
      </div>
      <div className="navbar-actions">
        {view !== 'upload' && (
          <button type="button" className="navbar-cta" onClick={onOpenUpload}>Get started</button>
        )}
      </div>
    </nav>
  )
}

function HomeView({ games, onSelect, onOpenUpload, loading }) {
  const totalFindings = games.reduce((sum, g) => sum + g.issue_count, 0)
  const totalPages = games.reduce((sum, g) => sum + g.pages, 0)

  return (
    <div className="home">
      <div className="hero">
        <h1>
          Stop the Inati.<br />
          <span className="h1-accent">Settle board game rule disputes with AI.</span>
        </h1>
        <p className="hero-name">
          <em>inat</em> (Albanian): the grudge that starts when someone reads the rules wrong.
        </p>
        <p className="subtitle">
          Upload your rulebook PDF and, in a minute or two, Inati flags passages that may be unclear or
          inconsistent, each with the quote, the page and a suggested rewrite. It only points to possible
          problems: the designer decides which ones are real.
        </p>
        <div className="hero-actions">
          <button type="button" className="hero-cta" onClick={onOpenUpload}>
            Upload Rulebook <span className="hero-cta-arrow">&rarr;</span>
          </button>
          <a
            href="#example"
            className="hero-cta-secondary"
            onClick={(e) => { e.preventDefault(); document.getElementById('example')?.scrollIntoView({ behavior: 'smooth' }) }}
          >
            View Demo Report
          </a>
        </div>
        {games.length > 0 && (
          <div className="hero-stats">
            <div className="hero-stat"><strong>{LANGUAGES.length}</strong><span>languages, for the rulebook and for the report</span></div>
            <div className="hero-stat">
              <strong>{Object.keys(CATEGORY_LABELS).length}</strong>
              <span>kinds of findings: wording, terms, timing, contradictions</span>
            </div>
            <div className="hero-stat">
              <strong>{totalFindings}</strong>
              <span>findings flagged in {games.length} example rulebooks ({totalPages} pages)</span>
            </div>
          </div>
        )}
      </div>

      <LiveExample onSelect={onSelect} />
      <HowItWorks />

      <section id="games">
        <h2>Already analysed</h2>
        <GameCarousel games={games} loading={loading} onSelect={onSelect} />
      </section>

      <EvaluationTrust />
      <Footer />
    </div>
  )
}

function ChatApp({ games, onSelectGame, onUpload, uploading, uploadError, chatKey, onNewChat, onExit }) {
  return (
    <div className="chat-app">
      <aside className="chat-sidebar">
        <div className="chat-sidebar-top">
          <button type="button" className="chat-exit" onClick={onExit}>&larr; Back to Inati</button>
          <button type="button" className="chat-new-btn" onClick={onNewChat}>+ New analysis</button>
        </div>
        <div className="chat-history">
          <span className="chat-history-label">Recent analyses</span>
          {games.length === 0 && <p className="chat-history-empty">Nothing analysed yet.</p>}
          {games.map((g) => (
            <button
              key={g.id}
              type="button"
              className="chat-history-item"
              title={displayName(g.source)}
              onClick={() => onSelectGame(g.id)}
            >
              <span className="chat-history-title">{cleanTitle(g.source)}</span>
              <span className="chat-history-meta">{plural(g.issue_count, 'finding')}</span>
            </button>
          ))}
        </div>
      </aside>
      <div className="chat-main">
        <ChatUpload key={chatKey} onUpload={onUpload} uploading={uploading} uploadError={uploadError} />
      </div>
    </div>
  )
}

export default function App() {
  const [games, setGames] = useState([])
  const [gamesLoading, setGamesLoading] = useState(true)
  const [result, setResult] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState(null)
  const [loadError, setLoadError] = useState(null)
  const [showChat, setShowChat] = useState(false)
  const [chatKey, setChatKey] = useState(0)

  const loadGames = () => {
    setGamesLoading(true)
    fetch(`${API_BASE}/games`)
      .then((r) => r.json())
      .then(setGames)
      .catch(() => setLoadError('Could not reach the backend API at ' + API_BASE + '. Is it running?'))
      .finally(() => setGamesLoading(false))
  }

  useEffect(loadGames, [])

  const selectGame = (id) => {
    fetch(`${API_BASE}/games/${encodeURIComponent(id)}`)
      .then((r) => r.json())
      .then((data) => setResult(normalizeResult(data)))
  }

  const uploadPdf = (file, model, { docLanguage, reportLanguage }) => {
    setUploading(true)
    setUploadError(null)
    const form = new FormData()
    form.append('file', file)
    form.append('model', model)
    form.append('doc_language', docLanguage)
    form.append('report_language', reportLanguage)
    fetch(`${API_BASE}/analyze`, { method: 'POST', body: form })
      .then(async (r) => {
        if (!r.ok) {
          const body = await r.json().catch(() => ({}))
          throw new Error(body.detail || `Request failed (${r.status})`)
        }
        return r.json()
      })
      .then((data) => {
        setResult(normalizeResult(data))
        loadGames()
      })
      .catch((e) => setUploadError(e.message))
      .finally(() => setUploading(false))
  }

  if (loadError) {
    return <div className="home"><Logo className="logo-lg" /><p className="warning">{loadError}</p></div>
  }

  return (
    <>
      <NavBar
        view={result ? 'report' : showChat ? 'upload' : 'home'}
        onHome={() => { setResult(null); setShowChat(false) }}
        onOpenUpload={() => { setResult(null); setShowChat(true) }}
      />
      {!result && showChat ? (
        <ChatApp
          games={games}
          onSelectGame={selectGame}
          onUpload={uploadPdf}
          uploading={uploading}
          uploadError={uploadError}
          chatKey={chatKey}
          onNewChat={() => setChatKey((k) => k + 1)}
          onExit={() => setShowChat(false)}
        />
      ) : (
        <div className="app">
        {result ? (
          <ResultsView result={result} onBack={() => setResult(null)} />
        ) : (
          <HomeView
            games={games}
            loading={gamesLoading}
            onSelect={selectGame}
            onOpenUpload={() => setShowChat(true)}
          />
        )}
        </div>
      )}
    </>
  )
}
