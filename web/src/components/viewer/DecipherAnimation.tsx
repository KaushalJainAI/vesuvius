import { useEffect, useMemo, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, BookOpen, Boxes, CheckCircle2, FileText, Image as ImageIcon, Play, RotateCcw, ScanLine } from 'lucide-react'
import type { SegmentMeta, AnimationPhase } from '@/types/segment'
import { useAnimationSequence } from '@/hooks/useAnimationSequence'

interface DecipherAnimationProps {
  segment: SegmentMeta
}

type DetectionResult = {
  n_lines?: number
  n_blobs?: number
  label_image?: { path?: string; width?: number; height?: number } | null
  lines?: Array<{ line_no: number; n_blobs: number; y_band: [number, number] }>
}

type SegmentResult = {
  n_strips?: number
  segment_text?: string
  segment_meaning?: string
  probable_scroll_summary?: string
  segment_summary?: {
    confidence?: string
    text?: string
    probable_summary?: string
    english_translation?: string
  }
  strips?: Array<{
    strip_id: number
    consensus?: { text?: string; characters?: Array<{ tier?: string }> }
    per_model?: Record<string, unknown>
  }>
}

const PHASES: AnimationPhase[] = ['ct', 'ink', 'letters', 'done']

const PHASE_META: Record<AnimationPhase, { roman: string; title: string; icon: typeof Activity }> = {
  idle: { roman: '-', title: 'Ready', icon: Play },
  ct: { roman: 'I', title: 'Volume context', icon: ScanLine },
  ink: { roman: 'II', title: 'Ink evidence', icon: ImageIcon },
  letters: { roman: 'III', title: 'Candidate alignment', icon: Boxes },
  done: { roman: 'IV', title: 'Reading review', icon: BookOpen },
}

function phaseDescription(phase: AnimationPhase, segment: SegmentMeta, detection: DetectionResult | null, result: SegmentResult | null): string {
  if (phase === 'ct') {
    return `${segment.layers} z-layers listed, ${segment.width.toLocaleString()} x ${segment.height.toLocaleString()} px surface.`
  }
  if (phase === 'ink') {
    const lines = detection?.n_lines ?? result?.n_strips ?? result?.strips?.length ?? 0
    const blobs = detection?.n_blobs ?? 0
    return lines
      ? `${lines} line bands isolated${blobs ? ` with ${blobs.toLocaleString()} ink components` : ''}.`
      : 'Enhanced label evidence checked for this record.'
  }
  if (phase === 'letters') {
    const chars = result?.strips?.flatMap(s => s.consensus?.characters ?? []).length ?? segment.letters.length
    return `${chars} candidate letter positions reviewed against the enhanced strip crops.`
  }
  if (phase === 'done') {
    const models = new Set<string>()
    result?.strips?.forEach(s => Object.keys(s.per_model ?? {}).forEach(k => models.add(k)))
    return `${models.size || 1} model reading${(models.size || 1) === 1 ? '' : 's'} compared; result remains provisional.`
  }
  return 'Press play to step through this segment record.'
}

function stripSrc(segmentId: string, stripId: number): string {
  return `/assets/decipher/${segmentId}/agent/strip_${String(stripId).padStart(2, '0')}_enhanced.png`
}

function compactReading(result: SegmentResult | null, segment: SegmentMeta): string {
  const text = result?.segment_summary?.text || result?.segment_text
  if (text?.trim()) return text.trim()
  return segment.letters.map(l => l.char).join(' ')
}

function readableConfidence(raw?: string): string {
  if (!raw) return 'provisional'
  if (raw === 'medium') return 'provisional'
  if (raw === 'low') return 'uncertain'
  return raw
}

function segmentSpecificNote(segment: SegmentMeta, detection: DetectionResult | null, result: SegmentResult | null): string {
  const lines = detection?.n_lines ?? result?.n_strips ?? result?.strips?.length ?? 0
  if (segment.id === '20231221180251') {
    return 'Recommended demo: complete segment image, buffered strip crops, and two model readings are available.'
  }
  if (!lines) return 'Archive record: detection metadata is available, but no stable strip transcription is prepared.'
  if (!segment.realLabel) return 'Research record: strip evidence is available, but no full label overlay is attached.'
  return `Research record: ${lines} prepared strip crops are available for inspection.`
}

export function DecipherAnimation({ segment }: DecipherAnimationProps) {
  const { phase, start, reset } = useAnimationSequence()
  const [detection, setDetection] = useState<DetectionResult | null>(null)
  const [result, setResult] = useState<SegmentResult | null>(null)
  const [fullImageOk, setFullImageOk] = useState(true)

  useEffect(() => {
    let cancelled = false
    setDetection(null)
    setResult(null)
    setFullImageOk(true)
    reset()

    fetch(`/assets/decipher/${segment.id}/result.detection.json`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (!cancelled) setDetection(d) })
      .catch(() => { if (!cancelled) setDetection(null) })

    fetch(`/assets/decipher/${segment.id}/result.json`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (!cancelled) setResult(d) })
      .catch(() => { if (!cancelled) setResult(null) })

    return () => { cancelled = true }
  }, [segment.id, reset])

  const phaseIdx = ['idle', ...PHASES].indexOf(phase)
  const activeMeta = PHASE_META[phase]
  const ActiveIcon = activeMeta.icon
  const strips = useMemo(() => {
    const prepared = result?.strips?.map(s => s.strip_id) ?? []
    return (prepared.length ? prepared : [0, 1, 2, 3, 4, 5]).slice(0, 6)
  }, [result])
  const reading = compactReading(result, segment)
  const confidence = readableConfidence(result?.segment_summary?.confidence)
  const high = result?.strips?.flatMap(s => s.consensus?.characters ?? []).filter(c => c.tier === 'HIGH').length ?? 0
  const provisional = result?.strips?.flatMap(s => s.consensus?.characters ?? []).filter(c => c.tier === 'MED').length ?? segment.letters.length

  return (
    <div className="flex flex-col gap-5">
      <div
        className="relative rounded-2xl overflow-hidden"
        style={{ minHeight: 500, background: '#0e0905', border: '1.5px solid rgba(200,184,138,0.2)', boxShadow: '0 8px 32px rgba(92,60,20,0.18)' }}
      >
        <div className="absolute inset-x-0 top-0 z-20 px-4 py-3 flex items-start justify-between gap-3"
             style={{ background: 'linear-gradient(180deg, rgba(0,0,0,0.82), rgba(0,0,0,0.15))' }}>
          <div className="flex items-center gap-3 min-w-0">
            <div className="w-10 h-10 rounded-full flex items-center justify-center font-serif font-bold"
                 style={{ background: 'var(--accent)', color: '#fff' }}>
              {phase === 'idle' ? <ActiveIcon size={17} /> : activeMeta.roman}
            </div>
            <div className="min-w-0">
              <p className="text-sm font-semibold" style={{ color: 'var(--gold-light)' }}>{activeMeta.title}</p>
              <p className="text-xs leading-relaxed" style={{ color: 'rgba(255,255,255,0.68)' }}>
                {phaseDescription(phase, segment, detection, result)}
              </p>
            </div>
          </div>
          <div className="hidden sm:flex flex-col items-end text-[10px] font-mono uppercase tracking-widest"
               style={{ color: 'rgba(212,168,80,0.78)' }}>
            <span>{segment.label}</span>
            <span>{segment.id}</span>
          </div>
        </div>

        <AnimatePresence mode="wait">
          {phase === 'idle' && (
            <motion.div
              key="idle"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 flex flex-col items-center justify-center px-6 text-center"
            >
              <FileText size={42} style={{ color: 'rgba(212,168,80,0.34)' }} />
              <h3 className="font-serif text-2xl mt-4 mb-2" style={{ color: 'rgba(255,255,255,0.78)' }}>
                Real processing record
              </h3>
              <p className="max-w-xl text-sm leading-relaxed" style={{ color: 'rgba(255,255,255,0.48)' }}>
                This sequence uses the selected segment's prepared assets: dimensions, enhanced label image,
                strip crops, candidate counts, and model reading status.
              </p>
              <p className="mt-3 text-xs font-mono" style={{ color: 'rgba(212,168,80,0.7)' }}>
                {segmentSpecificNote(segment, detection, result)}
              </p>
            </motion.div>
          )}

          {phase === 'ct' && (
            <motion.div key="ct" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 pt-20 p-6">
              <div className="h-full grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4">
                <div className="rounded-xl overflow-hidden relative" style={{ background: '#17120b', border: '1px solid rgba(200,184,138,0.18)' }}>
                  <img src={segment.ctPreview} alt="CT preview" className="w-full h-full object-cover opacity-75" onError={e => { e.currentTarget.style.opacity = '0' }} />
                  <div className="absolute inset-0" style={{ background: 'radial-gradient(circle at 50% 45%, rgba(202,161,90,0.18), transparent 58%)' }} />
                  <motion.div
                    className="absolute left-0 right-0 h-1"
                    initial={{ top: '10%' }}
                    animate={{ top: '88%' }}
                    transition={{ duration: 2.1, ease: 'linear' }}
                    style={{ background: 'linear-gradient(90deg, transparent, rgba(127,210,255,0.55), transparent)' }}
                  />
                </div>
                <ProcessStats segment={segment} detection={detection} result={result} phase={phase} />
              </div>
            </motion.div>
          )}

          {phase === 'ink' && (
            <motion.div key="ink" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 pt-20 p-6">
              <div className="h-full grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4">
                <div className="rounded-xl overflow-hidden relative flex items-center justify-center" style={{ background: '#000', border: '1px solid rgba(200,184,138,0.18)' }}>
                  {fullImageOk && (
                    <img
                      src={`/assets/decipher/${segment.id}/label_full.png`}
                      alt="full enhanced label"
                      className="max-w-full max-h-full object-contain"
                      onError={() => setFullImageOk(false)}
                    />
                  )}
                  {!fullImageOk && <img src={segment.inkProb} alt="ink probability" className="w-full h-full object-cover opacity-75" />}
                  {detection?.lines?.slice(0, 8).map((line, i) => (
                    <motion.div
                      key={line.line_no}
                      initial={{ opacity: 0, scaleX: 0.3 }}
                      animate={{ opacity: 1, scaleX: 1 }}
                      transition={{ delay: i * 0.08 }}
                      className="absolute left-[12%] right-[12%] h-px origin-left"
                      style={{ top: `${Math.min(88, Math.max(12, (line.y_band[0] / Math.max(1, segment.height)) * 100))}%`, background: 'rgba(255,188,74,0.72)' }}
                    />
                  ))}
                </div>
                <ProcessStats segment={segment} detection={detection} result={result} phase={phase} />
              </div>
            </motion.div>
          )}

          {phase === 'letters' && (
            <motion.div key="letters" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 pt-20 p-6">
              <div className="h-full grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4">
                <div className="rounded-xl p-3 overflow-hidden" style={{ background: '#050504', border: '1px solid rgba(200,184,138,0.18)' }}>
                  <div className="h-full flex flex-col gap-2 justify-center">
                    {strips.map((stripId, i) => (
                      <motion.div
                        key={stripId}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.12 }}
                        className="relative rounded overflow-hidden"
                        style={{ height: 54, background: '#000', border: '1px solid rgba(200,184,138,0.16)' }}
                      >
                        <img src={stripSrc(segment.id, stripId)} alt={`strip ${stripId}`} className="w-full h-full object-cover" onError={e => { e.currentTarget.style.opacity = '0' }} />
                        <span className="absolute top-1 left-1 px-1 rounded text-[9px] font-mono" style={{ background: 'rgba(0,0,0,0.72)', color: 'var(--gold-light)' }}>
                          L{stripId + 1}
                        </span>
                      </motion.div>
                    ))}
                  </div>
                </div>
                <ProcessStats segment={segment} detection={detection} result={result} phase={phase} />
              </div>
            </motion.div>
          )}

          {phase === 'done' && (
            <motion.div key="done" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="absolute inset-0 pt-20 p-6">
              <div className="h-full grid grid-cols-1 md:grid-cols-[1fr_260px] gap-4">
                <div className="rounded-xl p-5 overflow-auto" style={{ background: 'rgba(245,237,216,0.96)', border: '1px solid rgba(200,184,138,0.35)' }}>
                  <p className="text-xs font-mono uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>
                    Segment reading review
                  </p>
                  <p className="font-serif text-xl leading-relaxed whitespace-pre-wrap break-words" style={{ color: 'var(--text)', letterSpacing: '0.08em' }}>
                    {reading}
                  </p>
                  <div className="mt-4 grid grid-cols-3 gap-2">
                    <MiniMetric label="High" value={high} tone="#1e6f3a" />
                    <MiniMetric label="Provisional" value={provisional} tone="#b9851a" />
                    <MiniMetric label="Band" value={confidence} tone="#8b1a1a" />
                  </div>
                  {(result?.segment_summary?.probable_summary || result?.probable_scroll_summary || result?.segment_meaning) && (
                    <p className="mt-4 text-sm font-serif leading-relaxed" style={{ color: 'var(--text-mid)' }}>
                      {result.segment_summary?.probable_summary || result.probable_scroll_summary || result.segment_meaning}
                    </p>
                  )}
                </div>
                <ProcessStats segment={segment} detection={detection} result={result} phase={phase} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <button
          onClick={phase === 'idle' || phase === 'done' ? start : reset}
          className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all active:scale-95"
          style={{ background: 'var(--accent)', color: '#fff', boxShadow: '0 2px 10px rgba(139,26,26,0.3)' }}
        >
          {phase === 'idle' || phase === 'done'
            ? <><Play size={14} fill="#fff" /> Play real sequence</>
            : <><RotateCcw size={14} /> Reset</>}
        </button>

        <div className="flex items-center gap-2">
          {PHASES.map((p, i) => {
            const meta = PHASE_META[p]
            const Icon = meta.icon
            const active = phaseIdx === i + 1
            const complete = phaseIdx > i + 1
            return (
              <div key={p} className="flex items-center gap-2">
                <div className="h-8 w-8 rounded-full flex items-center justify-center"
                     style={{
                       background: complete ? 'var(--accent)' : active ? 'rgba(139,26,26,0.14)' : 'rgba(92,60,20,0.08)',
                       color: complete ? '#fff' : active ? 'var(--accent)' : 'var(--text-muted)',
                       border: '1px solid var(--border-light)',
                     }}
                     title={meta.title}>
                  {complete ? <CheckCircle2 size={14} /> : <Icon size={14} />}
                </div>
              </div>
            )
          })}
          <span className="text-xs ml-1 font-mono" style={{ color: 'var(--text-muted)' }}>{Math.max(0, phaseIdx)}/4</span>
        </div>
      </div>
    </div>
  )
}

function ProcessStats({
  segment, detection, result, phase,
}: {
  segment: SegmentMeta
  detection: DetectionResult | null
  result: SegmentResult | null
  phase: AnimationPhase
}) {
  const chars = result?.strips?.flatMap(s => s.consensus?.characters ?? []).length ?? segment.letters.length
  const models = new Set<string>()
  result?.strips?.forEach(s => Object.keys(s.per_model ?? {}).forEach(k => models.add(k)))
  const stats = [
    { label: 'Surface', value: `${segment.width.toLocaleString()} x ${segment.height.toLocaleString()}` },
    { label: 'Z-layers', value: String(segment.layers) },
    { label: 'Line bands', value: String(detection?.n_lines ?? result?.n_strips ?? result?.strips?.length ?? 0) },
    { label: 'Ink blobs', value: detection?.n_blobs ? detection.n_blobs.toLocaleString() : '-' },
    { label: 'Candidates', value: String(chars) },
    { label: 'Models', value: String(models.size || (result ? 1 : '-')) },
  ]

  return (
    <div className="rounded-xl p-4 overflow-hidden"
         style={{ background: 'rgba(245,237,216,0.96)', border: '1px solid rgba(200,184,138,0.35)' }}>
      <p className="text-xs font-mono uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>
        Segment-specific data
      </p>
      <div className="grid grid-cols-2 gap-2">
        {stats.map((s, i) => (
          <motion.div
            key={s.label}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.04 }}
            className="rounded-md p-2"
            style={{ background: 'rgba(92,60,20,0.055)', border: '1px solid var(--border-light)' }}
          >
            <p className="text-[10px] font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>{s.label}</p>
            <p className="font-serif font-semibold text-sm" style={{ color: 'var(--text)' }}>{s.value}</p>
          </motion.div>
        ))}
      </div>
      <div className="mt-4 rounded-md p-3" style={{ background: 'rgba(139,26,26,0.06)', border: '1px solid rgba(139,26,26,0.12)' }}>
        <p className="text-[10px] font-mono uppercase tracking-widest mb-1" style={{ color: 'var(--accent)' }}>
          Current step
        </p>
        <p className="text-sm leading-relaxed" style={{ color: 'var(--text-mid)' }}>
          {phaseDescription(phase, segment, detection, result)}
        </p>
      </div>
    </div>
  )
}

function MiniMetric({ label, value, tone }: { label: string; value: string | number; tone: string }) {
  return (
    <div className="rounded-md p-2" style={{ background: `${tone}10`, border: `1px solid ${tone}33` }}>
      <p className="text-[10px] font-mono uppercase tracking-widest" style={{ color: tone }}>{label}</p>
      <p className="font-serif font-bold" style={{ color: 'var(--text)' }}>{value}</p>
    </div>
  )
}
