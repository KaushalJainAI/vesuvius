import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Database, FileText, Layers, Microscope } from 'lucide-react'
import { GreekLetterBg } from '@/components/artifacts/GreekWatermark'
import { ScrollSVG } from '@/components/artifacts/ScrollSVG'
import { Skeleton } from '@/components/ui/Skeleton'
import { useManifest } from '@/hooks/useManifest'
import { formatBytes } from '@/lib/utils'
import type { Manifest } from '@/types/segment'

const DEMO_SEGMENT_ID = '20231221180251'

const REVIEW_METHOD = [
  { roman: 'I', icon: Microscope, title: 'Volume survey', desc: 'Review the available CT preview and segment dimensions before selecting a record.' },
  { roman: 'II', icon: Database, title: 'Ink evidence', desc: 'Inspect the enhanced probability image and label overlay included with the prepared record.' },
  { roman: 'III', icon: Layers, title: 'Candidate marks', desc: 'Compare extracted character candidates with their recorded confidence values.' },
  { roman: 'IV', icon: FileText, title: 'Reading note', desc: 'Review provisional model-assisted text separately from verified catalogue metadata.' },
]

function maxLettersForScale(manifest: Manifest | null): number {
  return Math.max(1, ...(manifest?.segments.map(seg => seg.letters.length) ?? [8]))
}

function segmentReviewNote(id: string): string {
  if (id === DEMO_SEGMENT_ID) return 'recommended demo'
  if (id === '20231016151002' || id === '20230702185753') return 'archive - weak visual crop'
  return 'research record'
}

export function Home() {
  const { manifest, loading } = useManifest()
  const navigate = useNavigate()
  const demoSegment = manifest?.segments.find(seg => seg.id === DEMO_SEGMENT_ID)
  const summary = useMemo(() => {
    const segments = manifest?.segments ?? []
    const totalSize = segments.reduce((sum, seg) => sum + seg.sizeMb, 0)
    const maxLayers = segments.reduce((max, seg) => Math.max(max, seg.layers), 0)
    const totalLetters = segments.reduce((sum, seg) => sum + seg.letters.length, 0)
    const labelled = segments.filter(seg => seg.realLabel).length

    return {
      rows: [
        { n: segments.length ? String(segments.length) : '11', label: 'segment records' },
        { n: totalSize ? formatBytes(totalSize) : '72 GB', label: 'prepared volume data' },
        { n: maxLayers ? String(maxLayers) : '33', label: 'z-layers listed' },
        { n: totalLetters ? String(totalLetters) : '88', label: 'candidate characters' },
      ],
      labelled,
      totalSize,
    }
  }, [manifest])

  return (
    <div className="min-h-screen pt-14">
      <section
        className="relative overflow-hidden torn-edge"
        style={{ background: 'linear-gradient(160deg, #F2E8D0 0%, #E9DBB8 70%, #F5EDD8 100%)', paddingBottom: 44 }}
      >
        <GreekLetterBg />
        <div className="max-w-6xl mx-auto px-4 sm:px-6 pt-12 pb-8">
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55 }}
            className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.2fr)_minmax(0,0.8fr)] gap-6 lg:gap-8 items-stretch"
          >
            <div className="card-papyrus rounded-lg p-6 lg:p-7">
              <div className="flex flex-wrap items-center gap-2 mb-5">
                <span className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                  PHercParis4
                </span>
                <span className="h-px w-8" style={{ background: 'var(--border)' }} />
                <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
                  Herculaneum papyrus segment catalogue
                </span>
              </div>

              <h1 className="font-serif text-3xl lg:text-4xl font-bold leading-tight mb-4" style={{ color: 'var(--text)' }}>
                PHercParis4 Segment Review
              </h1>
              <p className="text-base leading-relaxed mb-6" style={{ color: 'var(--text-mid)', maxWidth: 660 }}>
                A research review workspace for prepared PHercParis4 segment records, enhanced ink evidence,
                and provisional Greek readings. Each reading is shown with uncertainty so visual evidence and
                interpretation stay separate.
              </p>

              <div className="grid grid-cols-2 sm:grid-cols-4 gap-px mb-6" style={{ background: 'var(--border-light)' }}>
                {summary.rows.map(({ n, label }) => (
                  <div key={label} className="px-3 py-3" style={{ background: 'var(--bg-elevated)' }}>
                    <p className="font-serif font-bold text-xl" style={{ color: 'var(--text)' }}>{n}</p>
                    <p className="text-[11px] font-mono uppercase tracking-wide" style={{ color: 'var(--text-muted)' }}>{label}</p>
                  </div>
                ))}
              </div>

              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => navigate(`/viewer/${DEMO_SEGMENT_ID}`)}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm transition-all active:scale-95"
                  style={{ background: 'var(--accent)', color: '#fff' }}
                >
                  Open evaluator demo <ArrowRight size={15} />
                </button>
                <button
                  onClick={() => document.getElementById('segments-section')?.scrollIntoView({ behavior: 'smooth' })}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-lg font-semibold text-sm transition-all active:scale-95"
                  style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-mid)', border: '1px solid var(--border-light)' }}
                >
                  Browse records
                </button>
              </div>
            </div>

            <div className="card-papyrus rounded-lg p-5 flex flex-col justify-between gap-5">
              <div>
                <p className="text-xs font-mono uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>
                  Manuscript preview
                </p>
                <div className="min-h-[210px] flex items-center justify-center rounded-md" style={{ background: 'rgba(92,60,20,0.06)' }}>
                  <ScrollSVG className="w-[190px] drop-shadow-xl" />
                </div>
              </div>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <div>
                  <dt className="text-xs font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Accession</dt>
                  <dd className="font-medium" style={{ color: 'var(--text)' }}>PHercParis4</dd>
                </div>
                <div>
                  <dt className="text-xs font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Status</dt>
                  <dd className="font-medium" style={{ color: 'var(--text)' }}>review set</dd>
                </div>
                <div>
                  <dt className="text-xs font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Labels</dt>
                  <dd className="font-medium" style={{ color: 'var(--text)' }}>{summary.labelled} available</dd>
                </div>
                <div>
                  <dt className="text-xs font-mono uppercase" style={{ color: 'var(--text-muted)' }}>Readings</dt>
                  <dd className="font-medium" style={{ color: 'var(--text)' }}>provisional</dd>
                </div>
              </dl>
            </div>
          </motion.div>
        </div>
      </section>

      <section className="py-16 px-6" style={{ background: 'var(--bg)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="mb-8">
            <p className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: 'var(--text-muted)' }}>Review Method</p>
            <h2 className="font-serif text-2xl font-bold" style={{ color: 'var(--text)' }}>How records are inspected</h2>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-px" style={{ background: 'var(--border-light)' }}>
            {REVIEW_METHOD.map((step, i) => (
              <motion.div
                key={step.roman}
                initial={{ opacity: 0, y: 12 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                transition={{ delay: i * 0.06 }}
                className="p-5"
                style={{ background: 'var(--bg-elevated)' }}
              >
                <div className="flex items-center justify-between mb-4">
                  <span className="font-serif font-bold" style={{ color: 'var(--accent)' }}>{step.roman}</span>
                  <step.icon size={18} style={{ color: 'var(--text-muted)' }} />
                </div>
                <p className="font-serif font-semibold text-base mb-2" style={{ color: 'var(--text)' }}>{step.title}</p>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{step.desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      <section id="segments-section" className="py-16 px-6" style={{ background: 'var(--bg)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 mb-6">
            <div>
              <p className="text-xs font-mono uppercase tracking-widest mb-1" style={{ color: 'var(--text-muted)' }}>Segment Catalogue</p>
              <h2 className="font-serif text-2xl font-bold" style={{ color: 'var(--text)' }}>
                {manifest?.segments.length ?? 11} PHercParis4 segment records
              </h2>
            </div>
            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>
              {manifest ? `~${formatBytes(summary.totalSize)}` : '~72 GB'} / prepared segment metadata
            </span>
          </div>

          {demoSegment && (
            <motion.button
              initial={{ opacity: 0, y: 8 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              onClick={() => navigate(`/viewer/${demoSegment.id}`)}
              className="group text-left rounded-lg p-5 mb-5 cursor-pointer transition-all active:scale-[0.995]"
              style={{ background: 'var(--bg-elevated)', border: '1.5px solid var(--accent)', boxShadow: '0 10px 30px rgba(92,60,20,0.10)' }}
            >
              <div className="flex flex-wrap items-start justify-between gap-3 mb-2">
                <div>
                  <p className="text-xs font-mono uppercase tracking-widest mb-1" style={{ color: 'var(--accent)' }}>Recommended evaluator segment</p>
                  <p className="font-serif font-bold text-xl" style={{ color: 'var(--text)' }}>{demoSegment.label}</p>
                  <p className="text-sm mt-1" style={{ color: 'var(--text-mid)' }}>
                    Best prepared record for review: complete segment evidence, six readable strip crops, and model comparison.
                  </p>
                </div>
                <ArrowRight size={18} className="group-hover:translate-x-0.5 transition-transform" style={{ color: 'var(--accent)' }} />
              </div>
              <div className="flex flex-wrap gap-2 text-[11px] font-mono">
                <span className="px-2 py-1 rounded" style={{ background: 'rgba(139,26,26,0.08)', color: 'var(--accent)' }}>79 candidate positions</span>
                <span className="px-2 py-1 rounded" style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-muted)' }}>2 model readings</span>
                <span className="px-2 py-1 rounded" style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-muted)' }}>speculative, not final</span>
              </div>
            </motion.button>
          )}

          <div className="grid grid-cols-1 gap-3">
            {loading
              ? Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-28 rounded-lg" />)
              : manifest?.segments.map((seg, i) => (
                <motion.button
                  key={seg.id}
                  initial={{ opacity: 0, y: 8 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: i * 0.03 }}
                  onClick={() => navigate(`/viewer/${seg.id}`)}
                  className="group text-left rounded-lg p-4 cursor-pointer transition-all active:scale-[0.995]"
                  style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)' }}
                >
                  <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)_auto] gap-4 items-start">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2 mb-1">
                        <p className="font-serif font-semibold text-lg break-words" style={{ color: 'var(--text)' }}>{seg.label}</p>
                        <span className="text-xs font-mono break-all" style={{ color: 'var(--text-muted)' }}>{seg.id}</span>
                      </div>
                      <p className="text-sm leading-snug" style={{ color: 'var(--text-mid)' }}>{seg.description}</p>
                    </div>

                    <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-2 gap-x-4 gap-y-2 text-xs min-w-0">
                      <span style={{ color: 'var(--text-muted)' }}><Database size={11} className="inline mr-1" />{formatBytes(seg.sizeMb)}</span>
                      <span style={{ color: 'var(--text-muted)' }}><Layers size={11} className="inline mr-1" />{seg.layers} layers</span>
                      <span style={{ color: 'var(--text-muted)' }}>{seg.width.toLocaleString()} x {seg.height.toLocaleString()} px</span>
                      <span style={{ color: 'var(--text-muted)' }}>{seg.letters.length} candidate chars</span>
                    </div>

                    <div className="flex lg:flex-col items-start lg:items-end gap-2">
                      <span className="text-[11px] font-mono px-2 py-1 rounded border" style={{ color: seg.realLabel ? 'var(--olive)' : 'var(--text-muted)', borderColor: 'var(--border-light)' }}>
                        {seg.realLabel ? 'label overlay available' : 'no label overlay'}
                      </span>
                      <span className="text-[11px] font-mono px-2 py-1 rounded border" style={{
                        color: seg.id === DEMO_SEGMENT_ID ? 'var(--accent)' : 'var(--text-muted)',
                        borderColor: 'var(--border-light)',
                      }}>
                        {segmentReviewNote(seg.id)}
                      </span>
                      <ArrowRight size={15} className="group-hover:translate-x-0.5 transition-transform mt-1" style={{ color: 'var(--text-muted)' }} />
                    </div>
                  </div>
                  <div className="mt-3 h-px" style={{ background: `linear-gradient(90deg, var(--accent) 0%, var(--accent) ${Math.min(100, (seg.letters.length / maxLettersForScale(manifest)) * 100)}%, var(--border-light) 0%)`, opacity: 0.45 }} />
                </motion.button>
              ))}
          </div>
        </div>
      </section>

      <footer className="py-8 px-6 text-center border-t" style={{ borderColor: 'var(--border-light)' }}>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          PHercParis4 prepared segment review interface
        </p>
      </footer>
    </div>
  )
}
