import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Scan, FlaskConical, ZoomIn, ZoomOut, X, BookText, Quote } from 'lucide-react'
import type { SegmentMeta } from '@/types/segment'

interface InteractiveDecoderProps {
  segment: SegmentMeta
}

type Tier = 'HIGH' | 'MED' | 'LOW'

interface ConsensusChar {
  char: string
  x_norm: number
  confidence: number
  tier: Tier
  alternatives: string[]
  votes: { model: string; char: string; confidence: number }[]
}

interface StripResult {
  strip_id: number
  image_path: string
  y_range: [number, number]
  x_range: [number, number]
  consensus: { text: string; characters: ConsensusChar[] }
  per_model?: Record<string, ModelReading>
}

interface ModelReading {
  display_name?: string
  parsed?: {
    line_text?: string
    translation_en?: string
    probable_summary?: string
    notes?: string
    overall_confidence?: string
  } | null
  error?: string | null
}

interface SegmentResult {
  seg_id: string
  n_strips: number
  mock?: boolean
  models_used_open_source: string[]
  segment_text?: string
  segment_meaning?: string
  segment_translation_en?: string
  probable_scroll_summary?: string
  segment_summary?: {
    text: string
    probable_summary: string
    english_translation?: string
    probable_scroll_summary?: string
    confidence: string
  }
  strips: StripResult[]
}

interface ScholarWord {
  greek: string
  english: string
  certainty: string
  x_norm_start?: number
  x_norm_end?: number
  confidence?: number
}

interface ScholarStrip {
  strip_id: number
  paraphrase_en?: string
  recognized_words?: ScholarWord[]
  word_divisions?: string[]
}

interface ScholarFile {
  strips?: ScholarStrip[]
  segment?: { overall_paraphrase?: string; probable_genre?: string }
}

const TIER_COLORS: Record<Tier, string> = {
  HIGH: '#3fcf6c',
  MED: '#e9a73b',
  LOW: '#e35a5a',
}

// Match the strip's natural ratio (3600 x 384 ≈ 9.4:1)
const STRIP_RATIO = '12 / 1'
const STRIP_BUFFER_X = '18px'
const STRIP_BUFFER_Y = '8px'

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/** Locate the [start,end] x_norm of a Greek substring inside consensus chars.
 *  Used when the scholar pipeline didn't emit explicit x_norm_start/end. */
function locateWord(word: string, chars: ConsensusChar[]): [number, number] | null {
  if (!word || chars.length === 0) return null
  const seq = chars.map(c => c.char).join('')
  const idx = seq.indexOf(word)
  if (idx < 0) return null
  const start = chars[idx]?.x_norm ?? 0
  const end = chars[idx + word.length - 1]?.x_norm ?? start
  return [Math.max(0, start - 0.005), Math.min(1, end + 0.005)]
}

function pctConfidence(v?: number): number {
  if (v == null || Number.isNaN(v)) return 0
  return Math.max(0, Math.min(100, Math.round(v * 100)))
}

function confidenceBucketColor(p: number): string {
  if (p >= 70) return TIER_COLORS.HIGH
  if (p >= 45) return TIER_COLORS.MED
  return TIER_COLORS.LOW
}

function stripDisplaySrc(segmentId: string, strip: StripResult): string {
  return `/assets/decipher/${segmentId}/agent/strip_${String(strip.strip_id).padStart(2, '0')}_enhanced.png`
}

function modelCount(strips: StripResult[], fallback: number): number {
  const slugs = new Set<string>()
  for (const strip of strips) {
    for (const slug of Object.keys(strip.per_model ?? {})) slugs.add(slug)
  }
  return Math.max(fallback, slugs.size)
}

function reviewBand(dataConfidence: string, hasClaude: boolean): string {
  if (hasClaude) return 'disputed'
  if (dataConfidence === 'medium') return 'provisional'
  if (dataConfidence === 'low') return 'uncertain'
  return dataConfidence || 'provisional'
}

// ---------------------------------------------------------------------------
// main component
// ---------------------------------------------------------------------------

export function InteractiveDecoder({ segment }: InteractiveDecoderProps) {
  const [data, setData] = useState<SegmentResult | null>(null)
  const [scholar, setScholar] = useState<ScholarFile | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [zoomedStrip, setZoomedStrip] = useState<number | null>(null)

  // -- load data ----------------------------------------------------------
  useEffect(() => {
    let cancelled = false
    setData(null); setScholar(null); setLoadError(null); setZoomedStrip(null)

    fetch(`/assets/decipher/${segment.id}/result.json`)
      .then(r => { if (!r.ok) throw new Error('no decipher result'); return r.json() })
      .then((d: SegmentResult) => {
        if (cancelled) return
        setData({
          ...d,
          strips: Array.isArray(d.strips) ? d.strips : [],
          models_used_open_source: Array.isArray(d.models_used_open_source) ? d.models_used_open_source : [],
          n_strips: d.n_strips ?? (Array.isArray(d.strips) ? d.strips.length : 0),
        })
      })
      .catch(e => { if (!cancelled) setLoadError(String(e)) })

    fetch(`/assets/decipher/${segment.id}/scholar.json`)
      .then(r => r.ok ? r.json() : null)
      .then((s: ScholarFile | null) => { if (!cancelled) setScholar(s) })
      .catch(() => { /* scholar is optional */ })

    return () => { cancelled = true }
  }, [segment.id])

  if (loadError) return <ErrorCard segmentId={segment.id} />
  if (!data) {
    return (
      <div className="rounded-2xl p-8 text-center"
           style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
        <p className="font-serif" style={{ color: 'var(--text-muted)' }}>Loading transcription data...</p>
      </div>
    )
  }
  if (!Array.isArray(data.strips) || data.strips.length === 0) {
    return (
      <div className="rounded-2xl p-6"
           style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
        <p className="font-serif font-semibold mb-1" style={{ color: 'var(--text)' }}>
          Transcription unavailable
        </p>
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
          This segment has no extracted text strips. Switch to the <span className="font-mono">Text Deciphering</span> view above.
        </p>
      </div>
    )
  }

  const allChars = data.strips.flatMap(s => s.consensus?.characters ?? [])
  const totalHi  = allChars.filter(c => c.tier === 'HIGH').length
  const totalMed = allChars.filter(c => c.tier === 'MED').length
  const totalLow = allChars.filter(c => c.tier === 'LOW').length
  const segmentText        = data.segment_summary?.text || data.segment_text || ''
  const segmentTranslation = data.segment_summary?.english_translation || data.segment_translation_en || ''
  const scrollSummary      = data.segment_summary?.probable_scroll_summary || data.probable_scroll_summary || ''
  const segmentConfidence  = data.segment_summary?.confidence || (allChars.length ? 'low' : 'none')
  const nModels = modelCount(data.strips, data.models_used_open_source.length)
  const hasClaude = data.strips.some(s => Object.keys(s.per_model ?? {}).some(slug => slug.includes('claude')))
  const displayedBand = reviewBand(segmentConfidence, hasClaude)
  const scholarStrips: Record<number, ScholarStrip> = {}
  for (const s of (scholar?.strips ?? [])) scholarStrips[s.strip_id] = s

  return (
    <div className="flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2.5">
          <div className="w-9 h-9 rounded-full flex items-center justify-center"
            style={{ background: 'var(--accent)', color: '#fff',
                     boxShadow: '0 0 12px rgba(139,26,26,0.5)' }}>
            <Scan size={16} />
          </div>
          <div>
            <p className="font-serif font-bold text-base" style={{ color: 'var(--text)' }}>
              Transcription workbench
            </p>
            <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
              {data.n_strips} text lines isolated on the CT surface - {allChars.length} candidate
              characters across {nModels} model readings -
              click any line to inspect
            </p>
          </div>
        </div>
      </div>

      {/* Mock notice */}
      {data.mock && (
        <div className="rounded-xl px-4 py-2.5 flex items-center gap-2.5"
             style={{ background: 'rgba(176, 0, 32, 0.08)', border: '1px solid rgba(176, 0, 32, 0.25)' }}>
          <FlaskConical size={14} style={{ color: '#b00020', flexShrink: 0 }} />
          <p className="text-xs" style={{ color: '#7a0016' }}>
            <span className="font-semibold">Fallback reading data:</span> ink strips are provisional prepared evidence;
            letters are provisional candidates. Set <span className="font-mono">OPENROUTER_API_KEY</span>
            {' '}in <span className="font-mono">.env</span> and rerun to refresh readings.
          </p>
        </div>
      )}

      {/* Tier counters */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Total chars', value: allChars.length, color: 'var(--text)' },
          { label: 'High conf.',  value: totalHi,         color: TIER_COLORS.HIGH },
          { label: 'Provisional', value: totalMed,        color: TIER_COLORS.MED },
          { label: 'Low',         value: totalLow,        color: TIER_COLORS.LOW },
        ].map(item => (
          <div key={item.label} className="rounded-xl px-4 py-3"
               style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
            <p className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>{item.label}</p>
            <p className="font-serif text-2xl font-bold" style={{ color: item.color }}>{item.value}</p>
          </div>
        ))}
      </div>

      {/* Manuscript: stacked CT-backed strips */}
      <div className="rounded-2xl overflow-hidden"
           style={{ background: '#0a0905', border: '1.5px solid rgba(200,184,138,0.2)',
                    boxShadow: '0 8px 32px rgba(92,60,20,0.18)' }}>
        <div className="px-4 py-2.5 flex items-center justify-between"
             style={{ borderBottom: '1px solid rgba(200,184,138,0.15)',
                      background: 'rgba(200,184,138,0.04)' }}>
          <span className="text-[10px] font-mono uppercase tracking-widest"
                style={{ color: 'rgba(212,168,80,0.75)' }}>
            Enhanced ink evidence - segment {segment.id}
          </span>
          <span className="text-[10px] font-mono" style={{ color: 'rgba(212,168,80,0.55)' }}>
            {data.strips.length} lines
          </span>
        </div>

        <div className="p-3 space-y-3">
          {data.strips.map(strip => (
            <StripRow
              key={strip.strip_id}
              segmentId={segment.id}
              strip={strip}
              scholar={scholarStrips[strip.strip_id]}
              onZoom={() => setZoomedStrip(strip.strip_id)}
            />
          ))}
        </div>
      </div>

      {/* Provisional reading + summary */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px] gap-4">
        <div className="rounded-2xl p-5"
             style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
          <p className="text-xs font-mono uppercase tracking-widest mb-3"
             style={{ color: 'var(--text-muted)' }}>
            Provisional reading - all lines
          </p>
          <div className="space-y-2">
            {data.strips.map(strip => (
              <div key={strip.strip_id} className="flex items-baseline gap-3">
                <span className="text-[10px] font-mono mt-1"
                      style={{ color: 'var(--text-muted)' }}>
                  L{strip.strip_id + 1}
                </span>
                <p className="font-serif text-xl break-all flex-1"
                   style={{ color: 'var(--text)', letterSpacing: '0.14em' }}>
                  {strip.consensus.characters.map((c, i) => (
                    <span key={i} style={{ color: TIER_COLORS[c.tier] }}>{c.char}</span>
                  ))}
                </p>
              </div>
            ))}
          </div>
        </div>

        <div className="rounded-2xl p-5"
             style={{ background: 'rgba(92,60,20,0.06)', border: '1px solid var(--border-light)' }}>
          <p className="text-xs font-mono uppercase tracking-widest mb-3"
             style={{ color: 'var(--text-muted)' }}>
            Reading confidence
          </p>
          {(['HIGH', 'MED', 'LOW'] as Tier[]).map(t => {
            const count = t === 'HIGH' ? totalHi : t === 'MED' ? totalMed : totalLow
            const label = t === 'HIGH' ? 'High agreement'
                        : t === 'MED' ? 'Partial agreement'
                        : 'Disputed / weak'
            return (
              <div key={t} className="flex items-center justify-between gap-3 text-sm mb-2 last:mb-0">
                <span className="flex items-center gap-2" style={{ color: 'var(--text-mid)' }}>
                  <span className="h-2.5 w-2.5 rounded-full" style={{ background: TIER_COLORS[t] }} />
                  <span className="text-xs">{label}</span>
                </span>
                <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>{count}</span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="rounded-2xl p-5"
           style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
        <div className="flex items-center justify-between gap-3 mb-3">
          <p className="text-xs font-mono uppercase tracking-widest"
             style={{ color: 'var(--text-muted)' }}>
            Probable segment contents
          </p>
          <span className="text-[10px] font-mono px-2 py-1 rounded"
                style={{
                  color: displayedBand === 'disputed' ? '#8b1a1a' : displayedBand === 'provisional' ? TIER_COLORS.MED : 'var(--text-muted)',
                  background: 'rgba(92,60,20,0.08)',
                }}>
            {displayedBand}
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-4">
          <div className="rounded-lg p-3"
               style={{ background: 'rgba(92,60,20,0.045)', border: '1px solid var(--border-light)' }}>
            <p className="text-[10px] uppercase font-mono tracking-widest mb-1"
               style={{ color: 'var(--text-muted)' }}>English translation</p>
            <p className="font-serif text-base leading-relaxed" style={{ color: 'var(--text)' }}>
              {segmentTranslation && segmentTranslation !== '[uncertain]' ? segmentTranslation : 'No stable English translation yet.'}
            </p>
          </div>
          <div className="rounded-lg p-3"
               style={{ background: 'rgba(92,60,20,0.045)', border: '1px solid var(--border-light)' }}>
            <p className="text-[10px] uppercase font-mono tracking-widest mb-1"
               style={{ color: 'var(--text-muted)' }}>Probable scroll summary</p>
            <p className="font-serif text-base leading-relaxed" style={{ color: 'var(--text)' }}>
              {scrollSummary || 'No stable segment-level meaning has been produced yet.'}
            </p>
          </div>
        </div>
        {segmentText && (
          <pre className="font-serif text-lg leading-relaxed whitespace-pre-wrap break-words p-3 rounded-lg"
               style={{
                 color: 'var(--text-mid)',
                 background: 'rgba(92,60,20,0.06)',
                 border: '1px solid var(--border-light)',
                 letterSpacing: '0.08em',
               }}>{segmentText}</pre>
        )}
      </div>

      {/* Zoom modal */}
      <AnimatePresence>
        {zoomedStrip !== null && (
            <ZoomModal
            segmentId={segment.id}
            strip={data.strips.find(s => s.strip_id === zoomedStrip)!}
            scholar={scholarStrips[zoomedStrip]}
            onClose={() => setZoomedStrip(null)}
          />
        )}
      </AnimatePresence>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Strip row (compact list view)
// ---------------------------------------------------------------------------

function StripRow({
  segmentId, strip, scholar, onZoom,
}: {
  segmentId: string
  strip: StripResult
  scholar?: ScholarStrip
  onZoom: () => void
}) {
  const chars = strip.consensus?.characters ?? []
  const words = resolveWords(scholar?.recognized_words, chars)

  return (
    <button
      onClick={onZoom}
      className="group relative w-full rounded-md overflow-hidden text-left"
      style={{
        aspectRatio: STRIP_RATIO,
        background: '#000',
        outline: '1px solid rgba(200,184,138,0.15)',
        cursor: 'zoom-in',
      }}
      aria-label={`Open line ${strip.strip_id + 1} in inspection view`}
    >
      <div
        className="absolute rounded overflow-hidden"
        style={{
          inset: `${STRIP_BUFFER_Y} ${STRIP_BUFFER_X}`,
          background: '#050504',
        }}
      >
        <img
          src={stripDisplaySrc(segmentId, strip)}
          alt={`enhanced ink evidence for line ${strip.strip_id + 1}`}
          className="absolute inset-0 w-full h-full object-cover"
          style={{ filter: 'contrast(1.08) brightness(0.96)' }}
          draggable={false}
          onError={e => {
            e.currentTarget.src = `/assets/decipher/${segmentId}/${strip.image_path}`
          }}
        />

        {/* Word boxes overlaid on top of ink */}
        {words.map((w, i) => {
          const left = w.x0 * 100
          const width = (w.x1 - w.x0) * 100
          const conf = pctConfidence(w.confidence)
          const col = confidenceBucketColor(conf)
          return (
            <div key={i}
                 className="absolute pointer-events-none"
                 style={{
                   left: `${left}%`,
                   width: `${width}%`,
                   top: '8%',
                   bottom: '8%',
                   border: `1.5px solid ${col}`,
                   background: `${col}10`,
                   borderRadius: 3,
                   boxShadow: `0 0 8px ${col}40`,
                 }}>
              <div className="absolute -top-3 left-0 px-1 rounded font-mono whitespace-nowrap overflow-hidden text-ellipsis"
                   style={{
                     background: 'rgba(0,0,0,0.78)',
                     color: col,
                     fontSize: 8,
                     lineHeight: '11px',
                     maxWidth: '100%',
                   }}>
                {w.greek}
              </div>
            </div>
          )
        })}
      </div>

      {/* Line label */}
      <div className="absolute top-1.5 left-1.5 px-1 py-0.5 rounded font-mono z-10"
           style={{ background: 'rgba(0,0,0,0.7)', color: 'rgba(212,168,80,0.95)', fontSize: 9, lineHeight: '11px' }}>
        L{strip.strip_id + 1}
      </div>

      {/* Hover hint */}
      <div className="absolute top-1.5 right-1.5 px-1 py-0.5 rounded font-mono z-10 flex items-center gap-1
                      opacity-0 group-hover:opacity-100 transition-opacity"
           style={{ background: 'rgba(0,0,0,0.7)', color: 'rgba(212,168,80,0.95)', fontSize: 9, lineHeight: '11px' }}>
        <ZoomIn size={9} /> inspect
      </div>

    </button>
  )
}

// ---------------------------------------------------------------------------
// Zoom modal
// ---------------------------------------------------------------------------

interface ResolvedWord {
  greek: string
  english: string
  certainty: string
  confidence?: number
  x0: number
  x1: number
}

function resolveWords(words: ScholarWord[] | undefined, chars: ConsensusChar[]): ResolvedWord[] {
  if (!words || words.length === 0) return []
  const out: ResolvedWord[] = []
  for (const w of words) {
    if (!w.greek) continue
    let x0 = w.x_norm_start ?? 0
    let x1 = w.x_norm_end ?? 0
    if (!(x1 > x0)) {
      const located = locateWord(w.greek, chars)
      if (located) { x0 = located[0]; x1 = located[1] }
    }
    if (x1 <= x0) continue
    out.push({
      greek: w.greek,
      english: w.english,
      certainty: w.certainty,
      confidence: w.confidence,
      x0, x1,
    })
  }
  return out
}

function ZoomModal({
  segmentId, strip, scholar, onClose,
}: {
  segmentId: string
  strip: StripResult
  scholar?: ScholarStrip
  onClose: () => void
}) {
  const [zoom, setZoom] = useState(1)            // 1..6
  const [panX, setPanX] = useState(0)            // 0..1 of strip width (center fraction)
  const containerRef = useRef<HTMLDivElement>(null)
  const dragRef = useRef<{ startX: number; startPan: number } | null>(null)

  const chars = strip.consensus?.characters ?? []
  const words = useMemo(() => resolveWords(scholar?.recognized_words, chars), [scholar, chars])

  // ESC to close
  useEffect(() => {
    const h = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [onClose])

  // Wheel zoom
  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault()
    setZoom(z => Math.max(1, Math.min(6, z * (e.deltaY < 0 ? 1.12 : 0.9))))
  }, [])

  // Drag pan
  const onMouseDown = useCallback((e: React.MouseEvent) => {
    if (zoom <= 1) return
    dragRef.current = { startX: e.clientX, startPan: panX }
  }, [zoom, panX])

  const onMouseMove = useCallback((e: React.MouseEvent) => {
    if (!dragRef.current || !containerRef.current) return
    const rect = containerRef.current.getBoundingClientRect()
    const dx = (e.clientX - dragRef.current.startX) / rect.width
    setPanX(Math.max(0, Math.min(1, dragRef.current.startPan - dx / zoom)))
  }, [zoom])

  const onMouseUp = useCallback(() => { dragRef.current = null }, [])

  // Translate so panX is at viewport centre
  const translatePct = (panX * zoom - 0.5 * (zoom - 1)) * 100

  const overallParaphrase = scholar?.paraphrase_en
  const charsHi = chars.filter(c => c.tier === 'HIGH').length
  const charsMed = chars.filter(c => c.tier === 'MED').length
  const charsLow = chars.filter(c => c.tier === 'LOW').length
  const modelReadings = Object.entries(strip.per_model ?? {})
    .map(([slug, entry]) => ({ slug, ...entry }))
    .filter(entry => entry.parsed || entry.error)

  return (
    <motion.div
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      className="fixed inset-0 z-50 flex items-center justify-center p-3 sm:p-6"
      style={{ background: 'rgba(14,9,5,0.88)', backdropFilter: 'blur(6px)' }}
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.96, y: 12 }} animate={{ scale: 1, y: 0 }} exit={{ scale: 0.96, y: 8 }}
        transition={{ duration: 0.18 }}
        onClick={e => e.stopPropagation()}
        className="relative w-full max-w-7xl max-h-[92vh] overflow-y-auto rounded-2xl"
        style={{ background: 'var(--bg)', border: '1px solid rgba(200,184,138,0.3)', boxShadow: '0 20px 60px rgba(0,0,0,0.6)' }}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 flex items-center justify-between px-5 py-3"
             style={{ background: 'rgba(245,237,216,0.95)', backdropFilter: 'blur(8px)',
                      borderBottom: '1px solid var(--border-light)' }}>
          <div>
            <p className="text-[10px] font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
              Line inspection · segment {segmentId}
            </p>
            <p className="font-serif font-bold text-lg" style={{ color: 'var(--text)' }}>
              Line L{strip.strip_id + 1}
              <span className="ml-3 text-xs font-mono font-normal" style={{ color: 'var(--text-muted)' }}>
                y {strip.y_range[0]}–{strip.y_range[1]} · x {strip.x_range[0]}–{strip.x_range[1]}
              </span>
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button onClick={() => setZoom(z => Math.max(1, z * 0.83))}
                    className="p-2 rounded-lg" style={{ background: 'rgba(92,60,20,0.08)' }} aria-label="Zoom out">
              <ZoomOut size={14} style={{ color: 'var(--text-mid)' }} />
            </button>
            <span className="text-xs font-mono w-10 text-center" style={{ color: 'var(--text-mid)' }}>
              {zoom.toFixed(1)}×
            </span>
            <button onClick={() => setZoom(z => Math.min(6, z * 1.2))}
                    className="p-2 rounded-lg" style={{ background: 'rgba(92,60,20,0.08)' }} aria-label="Zoom in">
              <ZoomIn size={14} style={{ color: 'var(--text-mid)' }} />
            </button>
            <button onClick={onClose}
                    className="ml-2 p-2 rounded-lg" style={{ background: 'rgba(139,26,26,0.12)' }} aria-label="Close">
              <X size={14} style={{ color: 'var(--accent)' }} />
            </button>
          </div>
        </div>

        {/* Enlarged strip viewport */}
        <div className="p-5 space-y-5">
          <div
            ref={containerRef}
            className="relative w-full overflow-hidden rounded-lg select-none"
            style={{
              aspectRatio: STRIP_RATIO,
              background: '#0a0905',
              border: '1px solid rgba(200,184,138,0.25)',
              cursor: zoom > 1 ? (dragRef.current ? 'grabbing' : 'grab') : 'default',
            }}
            onWheel={onWheel}
            onMouseDown={onMouseDown}
            onMouseMove={onMouseMove}
            onMouseUp={onMouseUp}
            onMouseLeave={onMouseUp}
          >
            <div
              className="absolute overflow-hidden rounded"
              style={{
                inset: `${STRIP_BUFFER_Y} ${STRIP_BUFFER_X}`,
                transform: `translateX(${-translatePct}%) scale(${zoom})`,
                transformOrigin: '0% 50%',
                transition: dragRef.current ? 'none' : 'transform 80ms',
                willChange: 'transform',
                background: '#050504',
              }}
            >
              <img
                src={stripDisplaySrc(segmentId, strip)}
                alt={`enhanced ink evidence for line ${strip.strip_id + 1}`}
                className="absolute inset-0 w-full h-full object-cover"
                draggable={false}
                style={{ filter: 'contrast(1.08) brightness(0.96)' }}
                onError={e => {
                  e.currentTarget.src = `/assets/decipher/${segmentId}/${strip.image_path}`
                }}
              />

              {/* Word boxes */}
              {words.map((w, i) => {
                const conf = pctConfidence(w.confidence)
                const col = confidenceBucketColor(conf)
                return (
                  <div key={i}
                       className="absolute pointer-events-none"
                       style={{
                         left: `${w.x0 * 100}%`,
                         width: `${(w.x1 - w.x0) * 100}%`,
                         top: '6%', bottom: '6%',
                         border: `1.5px solid ${col}`,
                         background: `${col}10`,
                         borderRadius: 3,
                         boxShadow: `0 0 12px ${col}50`,
                       }}>
                    <div className="absolute -top-4 left-0 px-1 py-0.5 rounded font-mono whitespace-nowrap"
                         style={{
                           background: 'rgba(0,0,0,0.85)',
                           color: col,
                           fontSize: Math.max(8, 10 / Math.sqrt(zoom)),
                           lineHeight: 1.1,
                           transformOrigin: '0% 100%',
                         }}>
                      {w.greek}
                    </div>
                  </div>
                )
              })}

              {/* Per-character markers — only shown when zoomed in enough */}
              {zoom >= 2 && chars.map((c, i) => (
                <div key={i}
                     className="absolute pointer-events-none font-serif font-bold"
                     style={{
                       left: `${c.x_norm * 100}%`,
                       bottom: '4%',
                       transform: `translate(-50%, 0) scale(${1 / Math.sqrt(zoom)})`,
                       transformOrigin: '50% 100%',
                       color: TIER_COLORS[c.tier],
                       fontSize: 10,
                       textShadow: `0 0 4px ${TIER_COLORS[c.tier]}88`,
                       opacity: 0.85,
                     }}>
                  {c.char}
                </div>
              ))}
            </div>

            {/* Hint */}
            {zoom === 1 && (
              <div className="absolute bottom-2 left-1/2 -translate-x-1/2 px-2 py-1 rounded text-[10px] font-mono"
                   style={{ background: 'rgba(0,0,0,0.6)', color: 'rgba(212,168,80,0.85)' }}>
                scroll to zoom · drag to pan when zoomed
              </div>
            )}
          </div>

          {/* Reading + words panel */}
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4">
            <div className="rounded-xl p-4"
                 style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
              <p className="text-[10px] font-mono uppercase tracking-widest mb-2"
                 style={{ color: 'var(--text-muted)' }}>Consensus reading</p>
              <p className="font-serif text-lg break-all mb-4"
                 style={{ color: 'var(--text)', letterSpacing: '0.12em' }}>
                {chars.map((c, i) => (
                  <span key={i} style={{ color: TIER_COLORS[c.tier] }}>{c.char}</span>
                ))}
              </p>

              {overallParaphrase && (
                <div className="rounded-lg p-3 mb-3"
                     style={{ background: 'rgba(92,60,20,0.06)', border: '1px solid var(--border-light)' }}>
                  <p className="text-[10px] font-mono uppercase tracking-widest mb-1 flex items-center gap-1.5"
                     style={{ color: 'var(--text-muted)' }}>
                    <Quote size={10} /> Probable meaning
                  </p>
                  <p className="font-serif text-sm leading-relaxed" style={{ color: 'var(--text)' }}>
                    {overallParaphrase}
                  </p>
                </div>
              )}

              <div className="flex gap-3 text-xs">
                <Counter color={TIER_COLORS.HIGH} label="high" value={charsHi} />
                <Counter color={TIER_COLORS.MED} label="med" value={charsMed} />
                <Counter color={TIER_COLORS.LOW} label="low" value={charsLow} />
              </div>
            </div>

            <div className="rounded-xl p-4"
                 style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
              <p className="text-[10px] font-mono uppercase tracking-widest mb-3 flex items-center gap-1.5"
                 style={{ color: 'var(--text-muted)' }}>
                <BookText size={10} /> Words read in this line
              </p>
              {words.length === 0 ? (
                <p className="text-xs italic" style={{ color: 'var(--text-muted)' }}>
                  No words identified in this line yet. The scholar pipeline produces
                  word-level readings when run with an API key.
                </p>
              ) : (
                <div className="space-y-3">
                  {words.map((w, i) => {
                    const conf = pctConfidence(w.confidence)
                    const col = confidenceBucketColor(conf)
                    return (
                      <div key={i}>
                        <div className="flex items-baseline justify-between gap-2 mb-1">
                          <span className="font-serif text-base font-semibold"
                                style={{ color: col, letterSpacing: '0.06em' }}>
                            {w.greek}
                          </span>
                          <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                            {conf}%
                          </span>
                        </div>
                        {w.english && (
                          <p className="text-xs italic mb-1" style={{ color: 'var(--text-mid)' }}>
                            {w.english}
                          </p>
                        )}
                        <div className="h-1 rounded-full overflow-hidden"
                             style={{ background: 'rgba(92,60,20,0.12)' }}>
                          <div className="h-full rounded-full"
                               style={{ width: `${conf}%`, background: col }} />
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>

          {modelReadings.length > 0 && (
            <div className="rounded-xl p-4"
                 style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
              <p className="text-[10px] font-mono uppercase tracking-widest mb-3"
                 style={{ color: 'var(--text-muted)' }}>
                Model comparison
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {modelReadings.map(entry => {
                  const parsed = entry.parsed
                  const isClaude = entry.slug.includes('claude')
                  const isManual = entry.slug.startsWith('manual/')
                  return (
                    <div key={entry.slug}
                         className="rounded-lg p-3"
                         style={{
                           background: isClaude ? 'rgba(185,133,26,0.10)' : 'rgba(92,60,20,0.045)',
                           border: `1px solid ${isClaude ? 'rgba(185,133,26,0.34)' : 'var(--border-light)'}`,
                         }}>
                      <div className="flex items-center justify-between gap-2 mb-2">
                        <p className="text-[10px] font-mono uppercase tracking-widest truncate"
                           style={{ color: isClaude ? '#7a5a14' : 'var(--text-muted)' }}>
                          {entry.display_name || entry.slug}
                        </p>
                        {isManual && (
                          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                                style={{ background: 'rgba(0,0,0,0.08)', color: 'var(--text-muted)' }}>
                            manual
                          </span>
                        )}
                      </div>
                      {parsed ? (
                        <div className="space-y-2">
                          <p className="font-serif text-lg break-all"
                             style={{ color: 'var(--text)', letterSpacing: '0.10em' }}>
                            {parsed.line_text || '[no text]'}
                          </p>
                          {(parsed.probable_summary || parsed.translation_en) && (
                            <p className="font-serif text-sm leading-relaxed"
                               style={{ color: 'var(--text-mid)' }}>
                              {parsed.probable_summary || parsed.translation_en}
                            </p>
                          )}
                          {parsed.notes && (
                            <p className="text-[11px] leading-relaxed"
                               style={{ color: 'var(--text-muted)' }}>
                              {parsed.notes}
                            </p>
                          )}
                        </div>
                      ) : (
                        <p className="text-xs" style={{ color: '#8b1a1a' }}>
                          {entry.error || 'No parsed reading returned.'}
                        </p>
                      )}
                    </div>
                  )
                })}
              </div>
            </div>
          )}
        </div>
      </motion.div>
    </motion.div>
  )
}

function Counter({ color, label, value }: { color: string; label: string; value: number }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="h-2 w-2 rounded-full" style={{ background: color }} />
      <span className="font-mono" style={{ color: 'var(--text-mid)' }}>{value}</span>
      <span className="font-mono" style={{ color: 'var(--text-muted)' }}>{label}</span>
    </div>
  )
}

function ErrorCard({ segmentId }: { segmentId: string }) {
  return (
    <div className="rounded-2xl p-6"
         style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}>
      <p className="font-serif font-semibold mb-2" style={{ color: 'var(--text)' }}>
        No transcription data yet for this segment
      </p>
      <p className="text-sm mb-3" style={{ color: 'var(--text-muted)' }}>
        Run the segment reading script to generate strip images and readings:
      </p>
      <p className="text-xs font-mono p-2 rounded"
         style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-mid)' }}>
        python scripts/decipher_all_segments.py --only {segmentId}
      </p>
    </div>
  )
}
