import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Boxes, Brain, Cpu, ImageIcon, Languages, ScrollText, Sparkles } from 'lucide-react'
import { Skeleton } from '@/components/ui/Skeleton'

type TranscriptionLine = {
  index: number
  char: string | null
  alternates: string[]
  confidence: number
  tier: string | null
  bbox_strip: number[] | null
  word_index: number
  is_word_boundary?: boolean
}

type WordMatch = { greek: string; english: string; score: number; topic?: string }

type WordCandidate = {
  word_index: number
  char_indices: number[]
  greek_letters: string
  matches: WordMatch[]
}

type RichStrip = {
  strip_id: number
  image_path?: string
  annotated_image_path?: string
  enhanced_image_path?: string
  letters_text: string
  n_letters: number
  mean_confidence: number
  transcription_lines: TranscriptionLine[]
  word_candidates: WordCandidate[]
  paraphrase_en: string
  interpretation: string
  caveats: string
}

type LexicalDetail = { greek: string; english: string; topic?: string; score?: number }

type RichReading = {
  seg_id: string
  genre: string
  candidate_authors: string[]
  lexical_signals: string[]
  lexical_signal_details: LexicalDetail[]
  historical_context: string
  confidence_band: string
  mean_confidence: number
  n_letters_total: number
  top_letters: string[]
  overall_paraphrase: string
  interpretation: string[]
  caveats: string[]
  strips: RichStrip[]
  source?: string
}

type ClaudeReading = {
  seg_id: string
  source: string
  status?: string
  message?: string
  availability?: Record<string, unknown>
  strips: Array<{
    strip_id: number
    letters_text?: string
    paraphrase_en?: string
    interpretation?: string
    caveats?: string
    word_candidates?: WordCandidate[]
    _error?: string
  }>
}

type Props = { segmentId: string }

const BAND_COLOR: Record<string, string> = {
  supported: '#1e6f3a',
  plausible: '#b9851a',
  speculative: '#8b1a1a',
}

const TIER_COLOR: Record<string, string> = {
  HIGH: '#1e6f3a',
  MED: '#b9851a',
  LOW: '#8b1a1a',
}

function confidenceColor(c: number): string {
  if (c >= 0.7) return '#1e6f3a'
  if (c >= 0.45) return '#b9851a'
  return '#8b1a1a'
}

export function AgentReadingPanel({ segmentId }: Props) {
  const [reading, setReading] = useState<RichReading | null>(null)
  const [claude, setClaude] = useState<ClaudeReading | null>(null)
  const [loading, setLoading] = useState(true)
  const [showRaw, setShowRaw] = useState<Set<number>>(new Set())
  const [showClaude, setShowClaude] = useState(true)
  const [showFullSegment, setShowFullSegment] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setReading(null)
    setClaude(null)
    setShowFullSegment(true)
    Promise.all([
      fetch(`/assets/decipher/${segmentId}/agent/reading.json`)
        .then(r => (r.ok ? r.json() : null))
        .catch(() => null),
      fetch(`/assets/decipher/${segmentId}/agent/claude_reading.json`)
        .then(r => (r.ok ? r.json() : null))
        .catch(() => null),
    ]).then(([rich, cla]) => {
      if (cancelled) return
      setReading(rich)
      setClaude(cla)
      setLoading(false)
    })
    return () => { cancelled = true }
  }, [segmentId])

  if (loading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-28 rounded-lg" style={{ background: 'rgba(92,60,20,0.10)' }} />
        <Skeleton className="h-40 rounded-lg" style={{ background: 'rgba(92,60,20,0.10)' }} />
      </div>
    )
  }

  if (!reading) {
    return (
      <div className="card-papyrus rounded-lg p-6 text-center">
        <Brain size={22} className="mx-auto mb-2" style={{ color: 'var(--text-muted)' }} />
        <p className="font-serif text-base" style={{ color: 'var(--text)' }}>Agent reading not yet generated</p>
        <p className="text-xs mt-1" style={{ color: 'var(--text-muted)' }}>
          Run <code className="font-mono">python scripts/run_agent_decoding.py --only {segmentId}</code>
        </p>
      </div>
    )
  }

  const bandColor = BAND_COLOR[reading.confidence_band] || '#8b1a1a'
  const claudeHasUsableReading = claude?.status === 'ok' && claude.strips?.some(s =>
    !s._error && Boolean(s.letters_text || s.paraphrase_en || s.interpretation)
  )

  return (
    <div className="space-y-4">
      {/* Header card: genre + multi-paragraph interpretation */}
      <motion.div
        initial={false}
        className="rounded-lg p-5"
        style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)', color: 'var(--text)' }}
      >
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <ScrollText size={16} style={{ color: 'var(--accent)' }} />
            <p className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
              Evaluator synthesis
            </p>
          </div>
          <span
            className="text-xs font-semibold px-2.5 py-1 rounded-full uppercase tracking-wide"
            style={{ background: `${bandColor}1a`, color: bandColor }}
          >
            {reading.confidence_band}
          </span>
        </div>

        <h3 className="font-serif text-xl font-bold mb-1" style={{ color: 'var(--text)' }}>
          {reading.genre}
        </h3>
        {reading.candidate_authors?.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-1 mb-3">
            {reading.candidate_authors.map(a => (
              <span key={a} className="text-xs font-medium px-2 py-0.5 rounded"
                    style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-mid)' }}>{a}</span>
            ))}
          </div>
        )}

        <p className="font-serif text-base leading-relaxed mb-3" style={{ color: 'var(--text)' }}>
          {reading.overall_paraphrase}
        </p>

        {showFullSegment && (
          <div className="mb-4">
            <p className="text-[10px] font-mono uppercase tracking-widest mb-2"
               style={{ color: 'var(--text-muted)' }}>
              Complete segment evidence
            </p>
            <div className="bg-black rounded p-2 border" style={{ borderColor: 'var(--border-light)' }}>
              <img
                src={`/assets/decipher/${segmentId}/label_full.png`}
                alt={`Complete enhanced segment ${segmentId}`}
                className="w-full rounded object-contain"
                style={{ maxHeight: 260 }}
                onError={() => setShowFullSegment(false)}
              />
            </div>
          </div>
        )}

        {reading.interpretation?.map((para, i) => (
          <p key={i} className="font-serif text-sm leading-relaxed mb-2"
             style={{ color: 'var(--text)' }}>{para}</p>
        ))}

        <div className="mt-3 pt-3 border-t flex flex-wrap items-center gap-3 text-xs"
             style={{ borderColor: 'var(--border-light)', color: 'var(--text-muted)' }}>
          <span><b style={{ color: 'var(--text)' }}>{reading.n_letters_total}</b> letters</span>
          <span><b style={{ color: 'var(--text)' }}>{Math.round(reading.mean_confidence * 100)}%</b> mean evidence score</span>
          {reading.top_letters?.length > 0 && (
            <span>most-anchored: <span className="font-serif text-base" style={{ color: 'var(--text)' }}>
              {reading.top_letters.slice(0, 6).join(' ')}
            </span></span>
          )}
        </div>

        {reading.lexical_signal_details?.length > 0 && (
          <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border-light)' }}>
            <p className="text-[10px] font-mono uppercase tracking-widest mb-2"
               style={{ color: 'var(--text-muted)' }}>Lexical signals</p>
            <div className="flex flex-wrap gap-1.5">
              {reading.lexical_signal_details.map((w, j) => (
                <span key={j} className="text-xs px-2 py-0.5 rounded font-serif"
                      title={`${w.english}${w.topic ? ' / ' + w.topic : ''}${w.score ? ' / score ' + w.score.toFixed(2) : ''}`}
                      style={{
                        background: w.topic === 'prior' ? 'rgba(92,60,20,0.08)' : 'rgba(185,133,26,0.16)',
                        color: w.topic === 'prior' ? 'var(--text-muted)' : '#7a5a14',
                      }}>
                  {w.greek}
                  <span className="ml-1 text-[10px]" style={{ color: 'var(--text-muted)' }}>{w.english}</span>
                </span>
              ))}
            </div>
          </div>
        )}

      </motion.div>

      {/* Claude comparison is shown only when the stored run contains usable readings. */}
      {claudeHasUsableReading && (
        <motion.div
          initial={false}
          className="rounded-lg p-4"
          style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}
        >
          <button
            onClick={() => setShowClaude(v => !v)}
            className="w-full flex items-center justify-between"
          >
            <div className="flex items-center gap-2">
              <Sparkles size={14} style={{ color: '#7a5a14' }} />
              <span className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                Claude parallel reading
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded" style={{
                background: claude.status === 'ok' ? 'rgba(30,111,58,0.12)' : 'rgba(139,26,26,0.10)',
                color: claude.status === 'ok' ? '#1e6f3a' : '#8b1a1a',
              }}>{claude.status || 'unknown'}</span>
            </div>
            <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{showClaude ? 'hide' : 'show'}</span>
          </button>

          {showClaude && (
            <div className="mt-3 text-xs">
              {claude.status !== 'ok' ? (
                        <p style={{ color: 'var(--text-mid)' }}>{claude.message || 'Claude pathway unavailable.'}</p>
              ) : (
                <div className="space-y-2">
                  {claude.strips?.map((s, i) => (
                    <div key={i} className="border-t pt-2" style={{ borderColor: 'var(--border-light)' }}>
                      <p className="font-mono uppercase tracking-widest text-[10px] mb-1"
                         style={{ color: 'var(--text-muted)' }}>Claude · Strip {String(s.strip_id).padStart(2, '0')}</p>
                      {s._error ? (
                        <p style={{ color: '#8b1a1a' }}>error: {s._error}</p>
                      ) : (
                        <>
                          {s.letters_text && (
                            <p className="font-serif text-sm" style={{ color: 'var(--text)' }}>{s.letters_text}</p>
                          )}
                          {s.paraphrase_en && (
                            <p className="italic mt-1" style={{ color: 'var(--text-mid)' }}>{s.paraphrase_en}</p>
                          )}
                          {s.interpretation && (
                            <p className="mt-1" style={{ color: 'var(--text-mid)' }}>{s.interpretation}</p>
                          )}
                        </>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}

      {/* Per-strip cards: annotated image + transcription table + word candidates */}
      <div className="space-y-3">
        <p className="text-xs font-mono uppercase tracking-widest px-1"
           style={{ color: 'var(--text-muted)' }}>Per-strip evidence — annotated images and transcription</p>
        {reading.strips.map(s => {
          const annotated = s.annotated_image_path ? `/assets/decipher/${segmentId}/${s.annotated_image_path}` : null
          const enhanced = s.enhanced_image_path ? `/assets/decipher/${segmentId}/${s.enhanced_image_path}` : null
          const rawShown = showRaw.has(s.strip_id)
          const toggle = () => setShowRaw(prev => {
            const next = new Set(prev)
            if (next.has(s.strip_id)) next.delete(s.strip_id)
            else next.add(s.strip_id)
            return next
          })
          return (
            <motion.div
              key={s.strip_id}
              initial={false}
              className="rounded-lg p-4"
              style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}
            >
              <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
                <div className="flex items-center gap-2">
                  <Boxes size={13} style={{ color: 'var(--text-muted)' }} />
                  <p className="text-xs font-mono uppercase tracking-widest"
                     style={{ color: 'var(--text-muted)' }}>Strip {String(s.strip_id).padStart(2, '0')}</p>
                </div>
                <div className="flex items-center gap-3 text-xs" style={{ color: 'var(--text-muted)' }}>
                  <span>{s.n_letters} letters</span>
                  <span>·</span>
                  <span style={{ color: confidenceColor(s.mean_confidence) }}>
                    {Math.round(s.mean_confidence * 100)}% mean conf
                  </span>
                  {enhanced && (
                    <button onClick={toggle} className="flex items-center gap-1 hover:underline">
                      <ImageIcon size={11} /> {rawShown ? 'annotated' : 'enhanced'}
                    </button>
                  )}
                </div>
              </div>

              {annotated && (
                <div className="mb-3 bg-black/5 rounded p-1.5 border" style={{ borderColor: 'var(--border-light)' }}>
                  <img
                    src={rawShown && enhanced ? enhanced : annotated}
                    alt={`Strip ${s.strip_id} ${rawShown ? 'enhanced' : 'annotated'}`}
                    className="w-full rounded object-contain"
                    style={{ maxHeight: 200 }}
                  />
                </div>
              )}

              {s.letters_text && (
                <div className="mb-3">
                  <p className="text-[10px] font-mono uppercase tracking-widest mb-1"
                     style={{ color: 'var(--text-muted)' }}>Letter sequence</p>
                  <p className="font-serif text-lg tracking-wide break-all" style={{ color: 'var(--text)' }}>
                    {s.letters_text}
                  </p>
                </div>
              )}

              {s.word_candidates?.some(w => w.matches?.length > 0) && (
                <div className="mb-3">
                  <p className="text-[10px] font-mono uppercase tracking-widest mb-1"
                     style={{ color: 'var(--text-muted)' }}>Lexicon match candidates</p>
                  <div className="space-y-1 text-xs">
                    {s.word_candidates.filter(w => w.matches.length > 0).map(w => (
                      <div key={w.word_index} className="flex items-baseline gap-2">
                        <span className="font-serif text-base" style={{ color: 'var(--text)' }}>{w.greek_letters}</span>
                        <span style={{ color: 'var(--text-muted)' }}>→</span>
                        <div className="flex flex-wrap gap-1.5">
                          {w.matches.map((m, j) => (
                            <span key={j} className="px-1.5 py-0.5 rounded font-serif"
                                  style={{
                                    background: 'rgba(185,133,26,0.10)',
                                    color: 'var(--text-mid)',
                                  }}>
                              {m.greek} <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                                {m.english} ({Math.round(m.score * 100)}%)
                              </span>
                            </span>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {s.transcription_lines?.length > 0 && (
                <details className="mb-3">
                  <summary className="text-[10px] font-mono uppercase tracking-widest cursor-pointer"
                           style={{ color: 'var(--text-muted)' }}>
                    Per-letter transcription ({s.transcription_lines.length} positions)
                  </summary>
                  <div className="mt-2 overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr style={{ color: 'var(--text-muted)' }}>
                          <th className="text-left font-mono font-normal py-1">#</th>
                          <th className="text-left font-mono font-normal py-1">Read</th>
                          <th className="text-left font-mono font-normal py-1">Alts</th>
                          <th className="text-left font-mono font-normal py-1">Conf</th>
                          <th className="text-left font-mono font-normal py-1">Tier</th>
                          <th className="text-left font-mono font-normal py-1">Word</th>
                        </tr>
                      </thead>
                      <tbody>
                        {s.transcription_lines.map(l => (
                          <tr key={l.index} className="border-t" style={{ borderColor: 'var(--border-light)' }}>
                            <td className="py-1 font-mono" style={{ color: 'var(--text-muted)' }}>{l.index}</td>
                            <td className="py-1 font-serif text-base font-bold" style={{ color: 'var(--text)' }}>
                              {l.char || '?'}
                              {l.is_word_boundary && <span style={{ color: 'var(--text-muted)' }}> ·</span>}
                            </td>
                            <td className="py-1 font-serif" style={{ color: 'var(--text-muted)' }}>
                              {l.alternates.join(' ') || '-'}
                            </td>
                            <td className="py-1">
                              <span className="font-mono" style={{ color: confidenceColor(l.confidence) }}>
                                {Math.round(l.confidence * 100)}%
                              </span>
                            </td>
                            <td className="py-1 font-mono text-[10px]"
                                style={{ color: TIER_COLOR[l.tier || ''] || 'var(--text-muted)' }}>
                              {l.tier || '-'}
                            </td>
                            <td className="py-1 font-mono text-[10px]" style={{ color: 'var(--text-muted)' }}>
                              w{l.word_index}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              )}

              {s.paraphrase_en && (
                <div className="flex items-start gap-2 mb-1">
                  <Languages size={13} className="mt-0.5 flex-shrink-0" style={{ color: 'var(--text-muted)' }} />
                  <p className="text-sm italic font-serif" style={{ color: 'var(--text)' }}>{s.paraphrase_en}</p>
                </div>
              )}
              {s.interpretation && (
                <p className="text-xs leading-relaxed mt-2" style={{ color: 'var(--text-mid)' }}>{s.interpretation}</p>
              )}
            </motion.div>
          )
        })}
      </div>

      {reading.source && (
        <p className="text-[10px] font-mono px-1" style={{ color: 'var(--text-muted)' }}>
          <Cpu size={10} className="inline-block mr-1" /> source: {reading.source}
        </p>
      )}
    </div>
  )
}
