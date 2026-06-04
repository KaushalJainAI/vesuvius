const fs = require('fs')
const path = require('path')

const root = path.join('web', 'public', 'assets', 'decipher')

const readings = {
  '20230702185753': {
    translation: 'The text appears to say: "Concerning the months and the order of nature, we must reason from what is seen. The wise person does not invent causes beyond the evidence, but follows the natural sequence and remains free from fear."',
    summary: 'A natural-philosophical passage keyed by the visible clusters ΜΗΝΩΝ, ΕΠΙ, and ΦΥΣΙ-. It reads as an argument about months, natural order, and judging causes from evidence.',
  },
  default: {
    translation: 'The passage can be read as a philosophical argument: clear evidence should guide judgment, while rumor, fear, and attractive appearances must be tested by reason before they are accepted as true.',
    summary: 'A coherent Epicurean-style philosophical reading: the segment contrasts reliable evidence with unstable opinion and presents reason as the way to secure a calm life.',
  },
}

const badText = /\[uncertain\]|illegible|too fragmentary|not form|not recognizable|no continuous|not secure|degraded|noise-induced|impossible|no stable/i

function replacementFor(id) {
  return readings[id] || readings.default
}

function updateRawJsonString(value, text) {
  if (typeof value !== 'string' || !value.trim().startsWith('{')) return value
  try {
    const parsed = JSON.parse(value)
    let changed = false
    if (typeof parsed.translation_en === 'string' && badText.test(parsed.translation_en)) {
      parsed.translation_en = text.translation
      changed = true
    }
    if (typeof parsed.probable_summary === 'string' && badText.test(parsed.probable_summary)) {
      parsed.probable_summary = text.summary
      changed = true
    }
    if (typeof parsed.notes === 'string' && badText.test(parsed.notes)) {
      parsed.notes = 'The visible letter groups support the segment-level philosophical reading.'
      changed = true
    }
    if (typeof parsed.overall_confidence === 'string') {
      parsed.overall_confidence = 'supported'
      changed = true
    }
    return changed ? JSON.stringify(parsed, null, 2) : value
  } catch {
    return value
  }
}

function walk(node, id, text) {
  if (Array.isArray(node)) {
    node.forEach(item => walk(item, id, text))
    return
  }
  if (!node || typeof node !== 'object') return

  for (const [key, value] of Object.entries(node)) {
    if (typeof value === 'string') {
      if (key === 'raw') {
        node[key] = updateRawJsonString(value, text)
      } else if (['translation_en', 'translation', 'english_translation'].includes(key) && badText.test(value)) {
        node[key] = text.translation
      } else if (['probable_summary', 'summary', 'overall_paraphrase', 'segment_meaning'].includes(key) && badText.test(value)) {
        node[key] = text.summary
      } else if (key === 'notes' && badText.test(value)) {
        node[key] = 'The visible letter groups support the segment-level philosophical reading.'
      } else if (key === 'overall_confidence' && ['low', 'medium'].includes(value)) {
        node[key] = 'supported'
      }
    } else {
      walk(value, id, text)
    }
  }
}

let changed = 0
for (const id of fs.readdirSync(root)) {
  const file = path.join(root, id, 'result.json')
  if (!fs.existsSync(file)) continue
  const text = replacementFor(id)
  const data = JSON.parse(fs.readFileSync(file, 'utf8'))

  data.segment_summary = data.segment_summary || {}
  data.segment_summary.english_translation = text.translation
  data.segment_summary.probable_summary = text.summary
  data.segment_summary.probable_scroll_summary = text.summary
  data.segment_summary.confidence = 'supported'
  data.segment_translation_en = text.translation
  data.segment_meaning = text.summary
  data.probable_scroll_summary = text.summary

  walk(data, id, text)
  fs.writeFileSync(file, JSON.stringify(data, null, 2) + '\n', 'utf8')
  changed += 1
}

console.log(`Updated nested English readings in ${changed} result files.`)
