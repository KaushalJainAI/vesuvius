import { useParams, Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Layers, Database, ExternalLink, ChevronLeft, ChevronRight } from 'lucide-react'
import { useManifest } from '@/hooks/useManifest'
import { DecipherAnimation } from '@/components/viewer/DecipherAnimation'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatBytes } from '@/lib/utils'

export function Viewer() {
  const { id } = useParams<{ id: string }>()
  const { manifest, loading } = useManifest()
  const navigate = useNavigate()

  const segment = manifest?.segments.find(s => s.id === id)
  const segIdx   = manifest?.segments.findIndex(s => s.id === id) ?? -1
  const prev     = segIdx > 0 ? manifest!.segments[segIdx - 1] : null
  const next     = segIdx < (manifest?.segments.length ?? 0) - 1 ? manifest!.segments[segIdx + 1] : null

  if (loading) {
    return (
      <div className="min-h-screen pt-20 px-6 max-w-5xl mx-auto space-y-4">
        <Skeleton className="h-8 w-48" style={{ background: 'rgba(92,60,20,0.1)' }} />
        <Skeleton className="h-[480px] rounded-2xl" style={{ background: 'rgba(92,60,20,0.1)' }} />
      </div>
    )
  }

  if (!segment) {
    return (
      <div className="min-h-screen pt-20 flex flex-col items-center justify-center gap-4">
        <p className="font-serif text-2xl" style={{ color: 'var(--text-muted)' }}>Segment not found</p>
        <Link to="/" className="text-sm underline" style={{ color: 'var(--text-muted)' }}>← Back</Link>
      </div>
    )
  }

  return (
    <div className="min-h-screen pt-16" style={{ background: 'var(--bg)' }}>
      {/* Top stripe */}
      <div className="sticky top-14 z-30 px-6 py-3 flex items-center justify-between"
        style={{ background: 'rgba(245,237,216,0.92)', backdropFilter: 'blur(12px)', borderBottom: '1px solid var(--border-light)' }}>
        <div className="flex items-center gap-3">
          <Link to="/" className="flex items-center gap-1.5 text-sm transition-colors"
            style={{ color: 'var(--text-muted)' }}>
            <ArrowLeft size={15} /> Segments
          </Link>
          <span style={{ color: 'var(--border)' }}>/</span>
          <span className="font-serif font-semibold text-sm" style={{ color: 'var(--text)' }}>{segment.label}</span>
          <span className="font-mono text-xs hidden sm:inline" style={{ color: 'var(--text-muted)' }}>{segment.id}</span>
        </div>

        {/* Prev / next */}
        <div className="flex gap-1">
          <button onClick={() => prev && navigate(`/viewer/${prev.id}`)} disabled={!prev}
            className="p-2 rounded-lg transition-all disabled:opacity-30"
            style={{ background: 'rgba(92,60,20,0.06)' }}>
            <ChevronLeft size={15} style={{ color: 'var(--text-mid)' }} />
          </button>
          <button onClick={() => next && navigate(`/viewer/${next.id}`)} disabled={!next}
            className="p-2 rounded-lg transition-all disabled:opacity-30"
            style={{ background: 'rgba(92,60,20,0.06)' }}>
            <ChevronRight size={15} style={{ color: 'var(--text-mid)' }} />
          </button>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-6 py-8">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mb-6">
          <h1 className="font-serif text-3xl font-bold mb-1" style={{ color: 'var(--text)' }}>{segment.label}</h1>
          <p className="text-sm" style={{ color: 'var(--text-muted)' }}>{segment.description}</p>
        </motion.div>

        <div className="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-6">

          {/* Main animation */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }}>
            <DecipherAnimation segment={segment} />
          </motion.div>

          {/* Sidebar */}
          <motion.aside
            initial={{ opacity: 0, x: 12 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: 0.12 }}
            className="space-y-4"
          >
            {/* Metadata */}
            <div className="card-papyrus rounded-2xl p-5">
              <p className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>Segment Info</p>
              <dl className="space-y-3 text-sm">
                {[
                  { icon: Database, label: 'Size',     val: formatBytes(segment.sizeMb) },
                  { icon: Layers,   label: 'Z-layers', val: `${segment.layers} slices` },
                  { icon: null,     label: 'Width',    val: `${segment.width.toLocaleString()} px` },
                  { icon: null,     label: 'Height',   val: `${segment.height.toLocaleString()} px` },
                  { icon: null,     label: 'Letters',  val: `${segment.letters.length} found` },
                ].map(({ icon: Icon, label, val }) => (
                  <div key={label} className="flex items-center justify-between">
                    <dt className="flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}>
                      {Icon && <Icon size={12} />} {label}
                    </dt>
                    <dd className="font-medium" style={{ color: 'var(--text)' }}>{val}</dd>
                  </div>
                ))}
              </dl>
            </div>

            {/* Pipeline steps */}
            <div className="card-papyrus rounded-2xl p-5">
              <p className="text-xs font-mono uppercase tracking-widest mb-4" style={{ color: 'var(--text-muted)' }}>Pipeline Steps</p>
              <ol className="space-y-3">
                {[
                  { roman: 'I',   label: 'CT Scan',          sub: '33-layer Z-stack' },
                  { roman: 'II',  label: 'Ink Detection',    sub: '3D CNN → prob map' },
                  { roman: 'III', label: 'Letter Isolation', sub: 'CC analysis' },
                  { roman: 'IV',  label: 'Transcription',    sub: 'Confidence ranking' },
                ].map(step => (
                  <li key={step.roman} className="flex items-start gap-3">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-serif font-bold"
                      style={{ background: 'rgba(139,26,26,0.1)', color: 'var(--accent)' }}>
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

            {/* Label image preview */}
            {segment.realLabel && (
              <div className="card-papyrus rounded-2xl overflow-hidden">
                <div className="px-5 py-3 border-b" style={{ borderColor: 'var(--border-light)' }}>
                  <p className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                    ink_labels.tif
                  </p>
                </div>
                <div className="bg-black p-2">
                  <img src={segment.realLabel} alt="Ink labels" className="w-full rounded-lg object-contain" style={{ maxHeight: 180 }} />
                </div>
                <div className="px-5 py-2">
                  <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Pseudo-labels from GP-winner model · uint8 0–255
                  </p>
                </div>
              </div>
            )}

            {/* External link */}
            <a
              href={`https://data.aws.ash2txt.org/samples/PHercParis4/segments/${segment.id}/`}
              target="_blank" rel="noreferrer"
              className="flex items-center justify-between card-papyrus rounded-2xl p-4 text-sm group transition-all"
            >
              <span style={{ color: 'var(--text-mid)' }}>Raw segment data</span>
              <ExternalLink size={13} style={{ color: 'var(--text-muted)' }} className="group-hover:translate-x-0.5 transition-transform" />
            </a>
          </motion.aside>
        </div>
      </div>
    </div>
  )
}
