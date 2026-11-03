// The explanations that the model writes refer to sentences by their internal numbers ("S35", "sentence 42"). Those
// numbers mean nothing to a reader of the report, so they are shown as letters. The sentence that is quoted on the card
// (the first of the finding's sentence_ids) is "Sentence A", the next sentence of the finding is "Sentence B", and so
// on. A sentence that only the explanation mentions gets the next free letter, in the order in which it appears.
// A block of sentences ("sentences 210-214") becomes "other sentences".

const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

// "S35", "sentence 35", "Sentences 35 and 36", "S304-305", "sentence 162/235"
const REFERENCE = /(?:\b[Ss]entences?\s+S?|\bS)(\d{1,5})(?:\s*(?:[-–,&/]|\band\b)\s*S?\d{1,5})*/g

const isRange = (text) => /\d\s*[-–]\s*S?\d/.test(text)

export function labelSentences(text, sentenceIds = []) {
  const source = String(text ?? '')
  const letters = new Map(sentenceIds.slice(0, LETTERS.length).map((id, i) => [id, LETTERS[i]]))
  for (const m of source.matchAll(REFERENCE)) {
    if (isRange(m[0])) continue
    for (const n of m[0].match(/\d+/g).map(Number)) {
      if (!letters.has(n) && letters.size < LETTERS.length) letters.set(n, LETTERS[letters.size])
    }
  }

  return source.replace(REFERENCE, (whole, _first, offset) => {
    const names = isRange(whole) ? [] : [...new Set(whole.match(/\d+/g).map((n) => letters.get(Number(n))).filter(Boolean))]
    let out
    if (names.length === 0) out = 'other sentences'
    else if (names.length === 1) out = `Sentence ${names[0]}`
    else out = `Sentences ${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
    const startsSentence = offset === 0 || /[.?!]\s*$/.test(source.slice(0, offset))
    return startsSentence ? out.charAt(0).toUpperCase() + out.slice(1) : out
  })
}
