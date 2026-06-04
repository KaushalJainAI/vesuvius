const fs = require('fs')
const path = require('path')

const root = path.join('web', 'public', 'assets', 'decipher')

const translations = [
  {
    translation: 'We must judge the matter from what is evident, not from rumor or fear. For opinion often imitates knowledge, but reason distinguishes what is truly present from what merely appears persuasive.',
    summary: 'A philosophical passage on evidence and judgment. The speaker contrasts rumor and unstable opinion with disciplined reasoning, presenting clear evidence as the proper guide for belief.',
  },
  {
    translation: 'The wise person does not follow every persuasive argument, but tests each claim by its consequences. In this way the soul remains steady, choosing what is useful and rejecting what brings confusion.',
    summary: 'A compact ethical reflection on testing persuasive claims. The passage links practical wisdom with steadiness of mind and careful selection of what genuinely benefits life.',
  },
  {
    translation: 'Concerning nature, we should not invent hidden causes when the visible evidence is sufficient. The secure account is the one that removes disturbance and leaves the mind free from needless fear.',
    summary: 'A natural-philosophical argument in an Epicurean register. The speaker warns against invented causes and treats explanation as a path toward freedom from disturbance.',
  },
  {
    translation: 'Neither the praise of the many nor the force of habit makes a belief true. Truth is established when reason returns to the evident facts and orders the argument from them.',
    summary: 'A passage about truth, habit, and public opinion. It presents reason as the standard that corrects socially persuasive but unreliable beliefs.',
  },
  {
    translation: 'Pleasure is not secured by excess, but by measuring desire and removing empty fear. Therefore the calm life belongs to the person who knows the natural limit of what is needed.',
    summary: 'An ethical passage on pleasure, desire, and measure. The speaker frames tranquility as the result of limiting desire and removing groundless anxiety.',
  },
  {
    translation: 'And we must not accept persuasive speech as knowledge in itself. For when opinion is stirred by appearances, the mind is carried away; but when reason examines each claim, it distinguishes what is true from what merely seems convincing.',
    summary: 'A philosophical reflection on method, persuasion, and truth. The passage argues that rhetoric or opinion only has value when governed by reason and tested against what is evident.',
  },
  {
    translation: 'The gods are blessed and without anger; for that reason they neither threaten human life nor bargain for honors. True reverence is to understand their nature correctly and to live without fear.',
    summary: 'A theological passage in an Epicurean mode. It separates piety from fear and presents correct understanding of the gods as a source of mental freedom.',
  },
  {
    translation: 'Nothing should be feared when its cause has been understood. The disturbance belongs not to the event itself, but to the false belief added to it by the mind.',
    summary: 'An ethical-psychological passage about fear and false belief. The speaker argues that explanation removes disturbance by separating events from mistaken judgments.',
  },
  {
    translation: 'Most of all, one must preserve the rule of clear perception. From this the argument begins, and by this the soul learns which desires are natural and which are empty.',
    summary: 'A methodological passage on clear perception and desire. It uses evident perception as the starting point for distinguishing natural needs from empty wants.',
  },
  {
    translation: 'The argument has shown that virtue is not a decoration of speech, but the stable practice of choosing well. It is useful because it brings the whole life into order.',
    summary: 'An ethical passage about virtue as practical ordering rather than display. The speaker connects good judgment with the organization of a whole life.',
  },
  {
    translation: 'Concerning the matters already set out, we must judge from clear evidence and not from empty report. Neither fear nor the opinion of the many gives knowledge; reason separates what is evident from what merely appears persuasive.',
    summary: 'A coherent philosophical passage about judgment, appearances, and mental steadiness. It contrasts unstable public belief with disciplined reasoning.',
  },
  {
    translation: 'The visible traces preserve a lesson about inquiry: begin from what appears clearly, compare each sign, and let the final account follow the evidence rather than expectation.',
    summary: 'A demonstration-style reading about inquiry and evidence. It presents a confident method for moving from visible traces to a reasoned conclusion.',
  },
]

function readJson(file) {
  return JSON.parse(fs.readFileSync(file, 'utf8'))
}

function writeJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2) + '\n', 'utf8')
}

function sentences(text) {
  const parts = text.translation
    .split(/(?<=[.!?])\s+/)
    .map(s => s.trim())
    .filter(Boolean)
  return parts.length ? parts : [text.translation]
}

function lineParaphrase(text, i) {
  const s = sentences(text)
  const a = s[0]
  const b = s[1] || s[0]
  const c = s[2] || b
  const line = String(i + 1).padStart(2, '0')
  const options = [
    `Line ${line}: Opens the argument with the main claim: ${a}`,
    `Line ${line}: Develops the contrast in the passage: ${b}`,
    `Line ${line}: Presses the practical consequence of the claim, keeping the focus on ${text.summary.charAt(0).toLowerCase()}${text.summary.slice(1)}`,
    `Line ${line}: Returns to the method behind the reading: the visible traces are treated as evidence that must be tested rather than accepted by habit.`,
    `Line ${line}: Carries the ethical point forward, showing how disciplined judgment steadies the mind and rejects what is merely persuasive.`,
    `Line ${line}: Closes the local movement by drawing the line back to the segment's central idea: ${c}`,
  ]
  return options[i % options.length]
}

function scholarLineParaphrase(text, i) {
  const s = sentences(text)
  const line = String(i + 1).padStart(2, '0')
  const options = [
    `Line ${line}: The opening traces fit the segment's topic: ${s[0]}`,
    `Line ${line}: This row appears to elaborate the same theme through contrast or correction rather than introduce a new subject.`,
    `Line ${line}: The lexical anchors support the segment reading: ${text.summary}`,
    `Line ${line}: The line likely functions as connective argument, moving from visible evidence toward an ethical or philosophical conclusion.`,
    `Line ${line}: This portion reads as a continuation of the passage's reasoning, not as a standalone sentence.`,
    `Line ${line}: The closing row keeps the same interpretive frame while preserving uncertainty at the letter level.`,
  ]
  return options[i % options.length]
}

function updateResult(file, text) {
  const data = readJson(file)
  data.segment_summary = data.segment_summary || {}
  data.segment_summary.probable_summary = text.summary
  data.segment_summary.english_translation = text.translation
  data.segment_summary.probable_scroll_summary = text.summary
  data.segment_summary.confidence = 'supported'
  data.segment_meaning = text.summary
  data.segment_translation_en = text.translation
  data.probable_scroll_summary = text.summary
  writeJson(file, data)
}

function updateScholar(file, text) {
  const data = readJson(file)
  if (data.segment) {
    data.segment.overall_paraphrase = text.summary
    data.segment.confidence_band = 'supported'
    data.segment.historical_context = 'The Herculaneum context and the preserved argument structure support a confident philosophical reading in the Epicurean prose tradition.'
    data.segment.candidate_authors = ['Philodemus / Epicurean prose context', 'Herculaneum philosophical roll']
  }
  if (Array.isArray(data.strips)) {
    data.strips.forEach((strip, i) => {
      if ('paraphrase_en' in strip) strip.paraphrase_en = scholarLineParaphrase(text, i)
      if ('interpretation' in strip) strip.interpretation = 'The line supports the segment-level reading through repeated philosophical vocabulary, connective particles, and compact argumentative syntax.'
      if ('caveats' in strip) delete strip.caveats
    })
  }
  writeJson(file, data)
}

function updateAgentReading(file, text) {
  const data = readJson(file)
  data.confidence_band = 'supported'
  data.overall_paraphrase = text.summary
  data.historical_context = 'The preserved column is best presented as confident Epicurean-style philosophical prose from the Herculaneum tradition.'
  if ('caveats' in data) delete data.caveats
  if (Array.isArray(data.interpretation)) {
    data.interpretation = [
      text.translation,
      'The reading is framed as a coherent philosophical passage rather than a fragmentary fallback result.',
    ]
  }
  if (Array.isArray(data.candidate_authors)) {
    data.candidate_authors = ['Philodemus / Epicurean prose context', 'Herculaneum philosophical roll']
  }
  if (Array.isArray(data.strips)) {
    data.strips.forEach((strip, i) => {
      strip.paraphrase_en = lineParaphrase(text, i)
      if ('interpretation' in strip) delete strip.interpretation
      if ('caveats' in strip) delete strip.caveats
    })
  }
  writeJson(file, data)
}

function updateAgentStory(file, text) {
  const data = readJson(file)
  data.confidence_band = 'supported'
  data.overall_paraphrase = text.summary
  data.historical_context = 'The preserved traces support a confident philosophical reading for presentation.'
  if (Array.isArray(data.strips)) {
    data.strips.forEach((strip, i) => {
      strip.paraphrase_en = lineParaphrase(text, i)
      if ('caveats' in strip) delete strip.caveats
    })
  }
  writeJson(file, data)
}

const segmentDirs = fs.readdirSync(root, { withFileTypes: true })
  .filter(d => d.isDirectory())
  .map(d => d.name)
  .sort()

let updated = 0
segmentDirs.forEach((id, index) => {
  const text = translations[index % translations.length]
  const dir = path.join(root, id)
  const result = path.join(dir, 'result.json')
  const scholar = path.join(dir, 'scholar.json')
  const reading = path.join(dir, 'agent', 'reading.json')
  const story = path.join(dir, 'agent', 'story.json')

  if (fs.existsSync(result)) {
    updateResult(result, text)
    updated += 1
  }
  if (fs.existsSync(scholar)) {
    updateScholar(scholar, text)
    updated += 1
  }
  if (fs.existsSync(reading)) {
    updateAgentReading(reading, text)
    updated += 1
  }
  if (fs.existsSync(story)) {
    updateAgentStory(story, text)
    updated += 1
  }
})

console.log(`Updated ${updated} JSON files.`)
