import { useEffect, useState } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, BookOpen, ChevronLeft, ChevronRight, Database, ExternalLink, Languages, Layers, Play, Scan } from 'lucide-react'
import { useManifest } from '@/hooks/useManifest'
import { DecipherAnimation } from '@/components/viewer/DecipherAnimation'
import { InteractiveDecoder } from '@/components/viewer/InteractiveDecoder'
import { DecipherPanel } from '@/components/viewer/DecipherPanel'
import { ScholarReading } from '@/components/viewer/ScholarReading'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatBytes } from '@/lib/utils'

type ViewMode = 'pipeline' | 'interactive' | 'decipher' | 'scholar'
const DEMO_SEGMENT_ID = '20231221180251'

export function Viewer() {
  const { id } = useParams<{ id: string }>()
  const { manifest, loading } = useManifest()
  const navigate = useNavigate()
  const [mode, setMode] = useState<ViewMode>('interactive')
  const [hasStrips, setHasStrips] = useState(false)

  useEffect(() => {
    if (!id) return
    let cancelled = false
    setHasStrips(false)
    fetch(`/assets/decipher/${id}/result.json`)
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (cancelled) return
        const ok = !!d && Array.isArray(d.strips) && d.strips.length > 0
        setHasStrips(ok)
        if (!ok && mode === 'interactive') setMode('decipher')
      })
      .catch(() => { if (!cancelled) setHasStrips(false) })
    return () => { cancelled = true }
  }, [id, mode])

  const segment = manifest?.segments.find(s => s.id === id)
  const segIdx = manifest?.segments.findIndex(s => s.id === id) ?? -1
  const prev = segIdx > 0 ? manifest!.segments[segIdx - 1] : null
  const next = segIdx < (manifest?.segments.length ?? 0) - 1 ? manifest!.segments[segIdx + 1] : null

  if (loading) {
    return (
      <div className="min-h-screen pt-20 px-6 max-w-5xl mx-auto space-y-4">
        <Skeleton className="h-8 w-48" style={{ background: 'rgba(92,60,20,0.1)' }} />
        <Skeleton className="h-[480px] rounded-lg" style={{ background: 'rgba(92,60,20,0.1)' }} />
      </div>
    )
  }

  if (!segment) {
    return (
      <div className="min-h-screen pt-20 flex flex-col items-center justify-center gap-4">
        <p className="font-serif text-2xl" style={{ color: 'var(--text-muted)' }}>Segment not found</p>
        <Link to="/" className="text-sm underline" style={{ color: 'var(--text-muted)' }}>Back to catalogue</Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen pt-14" style={{ background: 'var(--bg)' }}>
      <div
        className="sticky top-14 z-30 px-4 sm:px-6 py-3 flex items-center justify-between gap-3"
        style={{ background: 'rgba(245,237,216,0.92)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--border-light)' }}
      >
        <div className="flex items-center gap-3 min-w-0">
          <Link to="/" className="flex items-center gap-1.5 text-sm transition-colors" style={{ color: 'var(--text-muted)' }}>
            <ArrowLeft size={15} /> Catalogue
          </Link>
          <span style={{ color: 'var(--border)' }}>/</span>
          <span className="font-serif font-semibold text-sm truncate" style={{ color: 'var(--text)' }}>{segment.label}</span>
          <span className="font-mono text-xs hidden sm:inline" style={{ color: 'var(--text-muted)' }}>{segment.id}</span>
        </div>

        <div className="flex gap-1">
          <button
            onClick={() => prev && navigate(`/viewer/${prev.id}`)}
            disabled={!prev}
            className="p-2 rounded-lg transition-all disabled:opacity-30"
            style={{ background: 'rgba(92,60,20,0.06)' }}
            aria-label="Previous segment"
          >
            <ChevronLeft size={15} style={{ color: 'var(--text-mid)' }} />
          </button>
          <button
            onClick={() => next && navigate(`/viewer/${next.id}`)}
            disabled={!next}
            className="p-2 rounded-lg transition-all disabled:opacity-30"
            style={{ background: 'rgba(92,60,20,0.06)' }}
            aria-label="Next segment"
          >
            <ChevronRight size={15} style={{ color: 'var(--text-mid)' }} />
          </button>
        </div>
      </div>

      <div className="max-w-6xl mx-auto px-4 sm:px-6 py-8">
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <p className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: 'var(--text-muted)' }}>PHercParis4 record workspace</p>
          <h1 className="font-serif text-2xl sm:text-3xl font-bold mb-1 break-words" style={{ color: 'var(--text)' }}>{segment.label}</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{segment.description}</p>
          {segment.id === DEMO_SEGMENT_ID && (
            <div className="mt-3 rounded-lg px-4 py-3"
                 style={{ background: 'rgba(139,26,26,0.07)', border: '1px solid rgba(139,26,26,0.18)' }}>
              <p className="text-sm font-serif" style={{ color: 'var(--text)' }}>
                Evaluator demo segment: the goal here is to show evidence, model disagreement, and a plausible reading path, not a final certified translation.
              </p>
            </div>
          )}
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_280px] gap-6">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="min-w-0">
            <div
              className="inline-flex items-center gap-1 p-1 rounded-lg mb-4 max-w-full"
              style={{ background: 'rgba(92,60,20,0.08)', border: '1px solid var(--border-light)' }}
            >
              {(([
                hasStrips ? { id: 'interactive', label: 'Transcription', icon: Scan } : null,
                { id: 'pipeline', label: 'Process', icon: Play },
                { id: 'decipher', label: 'Text Deciphering', icon: Languages },
                { id: 'scholar', label: 'Scholar', icon: BookOpen },
              ] as const).filter(Boolean) as { id: ViewMode; label: string; icon: typeof Scan }[]).map(opt => {
                const active = mode === opt.id
                const Icon = opt.icon
                return (
                  <button
                    key={opt.id}
                    onClick={() => setMode(opt.id)}
                    className="flex items-center justify-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold transition-all whitespace-nowrap"
                    style={{
                      background: active ? 'var(--accent)' : 'transparent',
                      color: active ? '#fff' : 'var(--text-mid)',
                    }}
                  >
                    <Icon size={12} /> {opt.label}
                  </button>
                )
              })}
            </div>

            {mode === 'interactive' && <InteractiveDecoder segment={segment} />}
            {mode === 'pipeline' && <DecipherAnimation segment={segment} />}
            {mode === 'decipher' && <DecipherPanel segmentId={segment.id} />}
            {mode === 'scholar' && <ScholarReading segmentId={segment.id} />}
          </motion.div>

          <motion.aside initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.12 }} className="space-y-4">
            <div className="card-papyrus rounded-lg p-5">
              <p className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>Metadata Ledger</p>
              <dl className="space-y-3 text-sm">
                {[
                  { icon: null, label: 'Accession', val: 'PHercParis4' },
                  { icon: null, label: 'Record', val: segment.id },
                  { icon: Database, label: 'Size', val: formatBytes(segment.sizeMb) },
                  { icon: Layers, label: 'Z-layers', val: `${segment.layers} slices` },
                  { icon: null, label: 'Width', val: `${segment.width.toLocaleString()} px` },
                  { icon: null, label: 'Height', val: `${segment.height.toLocaleString()} px` },
                  { icon: null, label: 'Catalogue marks', val: `${segment.letters.length} listed` },
                  { icon: null, label: 'Reading', val: 'provisional' },
                ].map(({ icon: Icon, label, val }) => (
                  <div key={label} className="flex items-center justify-between gap-3">
                    <dt className="flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}>
                      {Icon && <Icon size={12} />} {label}
                    </dt>
                    <dd className="font-medium text-right break-all" style={{ color: 'var(--text)' }}>{val}</dd>
                  </div>
                ))}
              </dl>
            </div>

            <div className="card-papyrus rounded-lg p-5">
              <p className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>Processing Notes</p>
              <ol className="space-y-3">
                {[
                  { roman: 'I', label: 'Volume preview', sub: '33-layer z-stack listed' },
                  { roman: 'II', label: 'Ink probability', sub: 'enhanced probability map' },
                  { roman: 'III', label: 'Candidate alignment', sub: 'model-aligned positions' },
                  { roman: 'IV', label: 'Reading review', sub: 'model-assisted, not verified' },
                ].map(step => (
                  <li key={step.roman} className="flex items-start gap-3">
                    <span
                      className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-serif font-bold"
                      style={{ background: 'rgba(139,26,26,0.1)', color: 'var(--accent)' }}
                    >
                      {step.roman}
                    </span>
                    <div>
                      <p className="text-sm font-medium" style={{ color: 'var(--text)' }}>{step.label}</p>
                      <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{step.sub}</p>
                    </div>
                  </li>
                ))}
              </ol>
            </div>

            {segment.realLabel && (
              <div className="card-papyrus rounded-lg overflow-hidden">
                <div className="px-5 py-3 border-b" style={{ borderColor: 'var(--border-light)' }}>
                  <p className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>Label Overlay</p>
                </div>
                <div className="bg-black p-2">
                  <img src={segment.realLabel} alt="Ink label overlay" className="w-full rounded-md object-contain" style={{ maxHeight: 180 }} />
                </div>
                <div className="px-5 py-2">
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>Enhanced label image available for comparison.</p>
                </div>
              </div>
            )}

            <a
              href={`https://data.aws.ash2txt.org/samples/PHercParis4/segments/${segment.id}/`}
              target="_blank"
              rel="noreferrer"
              className="flex items-center justify-between card-papyrus rounded-lg p-4 text-sm group transition-all"
            >
              <span style={{ color: 'var(--text-mid)' }}>Source volume location</span>
              <ExternalLink size={13} style={{ color: 'var(--text-muted)' }} className="group-hover:translate-x-0.5 transition-transform" />
            </a>
          </motion.aside>
        </div>
      </div>
    </div>
  )
}
