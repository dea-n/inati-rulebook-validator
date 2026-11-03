// Saving a report from the browser as a PDF: a printable page with the issues, the FAQ and the contradiction candidates.
// It is built from the result that is already loaded, so no request goes to the server and nothing leaves the computer.
// `labels` comes from App.jsx: { category(c), model(id), title(result) }.
import { labelSentences } from './sentenceRefs.js'

const esc = (s) =>
  String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')

const fileBase = (result, labels) => labels.title(result).replace(/[^\w.-]+/g, '_')
const pagesText = (pages) => `page${pages.length > 1 ? 's' : ''} ${pages.join(', ')}`

function download(filename, mime, text) {
  const url = URL.createObjectURL(new Blob([text], { type: `${mime};charset=utf-8` }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

// ---------------------------------------------------------------- printable report: all three parts, white page
const REPORT_CSS = `
  @page { size: A4; margin: 18mm; }
  body { font: 11pt/1.45 Georgia, 'Times New Roman', serif; color: #18181b; max-width: 820px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 20pt; margin: 0 0 4px; }
  h2 { font-size: 14pt; margin: 26px 0 8px; padding-bottom: 4px; border-bottom: 1px solid #d4d4d8; }
  .meta { color: #52525b; font-size: 9.5pt; margin: 0 0 4px; }
  .note { background: #f4f4f5; border-left: 3px solid #b45309; padding: 8px 12px; font-size: 10pt; margin: 10px 0 14px; }
  .item { break-inside: avoid; margin: 0 0 14px; padding-bottom: 10px; border-bottom: 1px solid #ececec; }
  .item p { margin: 4px 0; }
  blockquote { margin: 6px 0; padding-left: 10px; border-left: 3px solid #b45309; font: 9.5pt/1.4 Menlo, Consolas, monospace; }
  .rewrite { background: #fdf6e7; padding: 6px 10px; border-radius: 4px; }
  .quote-label { display: block; margin-bottom: 3px; font: 700 8pt/1.2 Georgia, 'Times New Roman', serif; letter-spacing: 0.06em; text-transform: uppercase; color: #52525b; }
  .warn { color: #c92a2a; font-size: 10pt; }
  .none { color: #52525b; font-style: italic; }
  footer { margin-top: 28px; color: #52525b; font-size: 9pt; }
`

export function reportHtml(result, labels, { autoPrint = true } = {}) {
  const issues = result.issues.filter((i) => i.category !== 'contradiction')
  const heading = (i) =>
    `<p class="meta"><b>${esc(labels.category(i.category))}</b>${i.severity ? ` · ${esc(i.severity)} severity` : ''} · ${esc(pagesText(i.pages))}</p>`
  const issue = (i) =>
    `<article class="item">${heading(i)}<blockquote>“${esc(i.quote)}”</blockquote><p>${esc(i.explanation)}</p>` +
    (i.suggested_rewrite ? `<p class="rewrite"><b>Suggested rewrite:</b> ${esc(i.suggested_rewrite)}</p>` : '') +
    (i.quote_verified ? '' : '<p class="warn">This quote could not be verified word for word against the source text.</p>') +
    '</article>'
  const faq = (e) =>
    `<article class="item"><p class="meta"><b>${esc(labels.category(e.category))}</b> · ${esc(e.confidence)} confidence · ${esc(pagesText(e.pages))}</p>` +
    `<p><b>Q:</b> ${esc(e.question)}</p><p><b>A:</b> ${esc(e.answer)}</p></article>`
  // A contradiction candidate, as in the interface: tagged "unverified" instead of with the severity, and the sentence
  // numbers of the explanation shown as letters (see sentenceRefs.js); only Sentence A is quoted.
  const sentencesOf = new Map(result.issues.map((i) => [i.id, i.sentence_ids || []]))
  const review = (r) => {
    const ids = sentencesOf.get(r.issue_id) || []
    const label = ids.length
      ? `<span class="quote-label">Sentence A${r.pages.length === 1 ? ` · page ${esc(r.pages[0])}` : ''}</span>`
      : ''
    return (
      `<article class="item"><p class="meta"><b>${esc(labels.category(r.category))}</b> · unverified · ${esc(pagesText(r.pages))}</p>` +
      `<blockquote>${label}“${esc(r.quote)}”</blockquote>` +
      (ids.length > 1 ? '<p class="meta">Only Sentence A is quoted. The other sentences are in the rulebook, on the pages named above.</p>' : '') +
      `<p>${esc(labelSentences(r.explanation, ids))}</p></article>`
    )
  }
  const list = (items, render) => (items.length ? items.map(render).join('') : '<p class="none">None.</p>')

  const analysed = String(result.analyzed_at || '').slice(0, 10)
  const model = result.model ? `${esc(labels.model(result.model))}` : 'an unknown model'
  // Only when a language other than English is involved; an English rulebook with an English report reads as before.
  const { document: docLanguage, report: reportLanguage } = result.language || {}
  const languages = docLanguage && !(docLanguage === 'English' && reportLanguage === 'English')
    ? `<p class="meta">Rulebook language: ${esc(docLanguage)} · explanations and FAQ in ${esc(reportLanguage)}. The quotes and the suggested rewrites are in the language of the rulebook.</p>`
    : ''
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>${esc(labels.title(result))} - Inati report</title><style>${REPORT_CSS}</style></head>
<body>
<h1>${esc(labels.title(result))}</h1>
<p class="meta">${esc(result.stats.pages)} pages · ${esc(result.stats.sentences)} sentences · analysed${analysed ? ` on ${esc(analysed)}` : ''} with ${model}</p>
${languages}<p class="meta">${issues.length} issue${issues.length === 1 ? '' : 's'} · ${result.faq.length} FAQ draft${result.faq.length === 1 ? '' : 's'} · ${result.needs_human_review.length} contradiction candidate${result.needs_human_review.length === 1 ? '' : 's'}</p>
<p class="note">These are suggestions for the author of the rulebook. They were written by a language model and have not been checked by a person.</p>

<h2>Unclear wording, terminology and timing (${issues.length})</h2>
${list(issues, issue)}

<h2>FAQ drafts (${result.faq.length})</h2>
<p class="note">Drafted by the model from the rulebook text. The confidence label is the model&rsquo;s own estimate. Check important rulings against the rulebook.</p>
${list(result.faq, faq)}

<h2>Contradiction candidates, to be checked (${result.needs_human_review.length})</h2>
<p class="note">In the evaluation of Inati, only 6% of the contradiction candidates were real (1 of 16), so each of these has to be checked by a person before it is used.</p>
${list(result.needs_human_review, review)}

<footer>Made with Inati on ${esc(new Date().toISOString().slice(0, 10))}.</footer>
${autoPrint ? '<script>window.addEventListener("load", function () { setTimeout(function () { window.print() }, 400) })</script>' : ''}
</body></html>`
}

// ---------------------------------------------------------------- the button
export function saveReportPdf(result, labels) {
  const html = reportHtml(result, labels)
  const url = URL.createObjectURL(new Blob([html], { type: 'text/html;charset=utf-8' }))
  // The page opens in a new tab and starts the print dialog, where "Save as PDF" is a choice of the browser.
  const tab = window.open(url, '_blank')
  if (!tab) download(`${fileBase(result, labels)}_report.html`, 'text/html', html) // pop-ups are blocked: save the page instead
  setTimeout(() => URL.revokeObjectURL(url), 60000)
}
