import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowRight, Layers, Database, Upload, BookOpen, Microscope, FileText } from 'lucide-react'
import { ScrollSVG } from '@/components/artifacts/ScrollSVG'
import { GreekLetterBg } from '@/components/artifacts/GreekWatermark'
import { UploadZone } from '@/components/upload/UploadZone'
import { Skeleton } from '@/components/ui/Skeleton'
import { useManifest } from '@/hooks/useManifest'
import { formatBytes } from '@/lib/utils'

const STEPS = [
  { roman: 'I',   icon: Microscope, title: 'CT Scan',        desc: '33 X-ray tomographic slices capture the compressed papyrus layers in 3D.' },
  { roman: 'II',  icon: Upload,     title: 'Ink Detection',  desc: 'A 3D convolutional network identifies ink deposits invisible to the naked eye.' },
  { roman: 'III', icon: BookOpen,   title: 'Letter Isolation', desc: 'Connected-component analysis extracts individual Greek letter candidates.' },
  { roman: 'IV',  icon: FileText,   title: 'Transcription',  desc: 'Letters are ranked by confidence and assembled into readable ancient text.' },
]

export function Home() {
  const { manifest, loading } = useManifest()
  const navigate = useNavigate()
  const [uploading, setUploading] = useState(false)
  const [uploadDone, setUploadDone] = useState(false)

  const handleUpload = (_files: { name: string; size: number; type: string }[]) => {
    setUploading(true)
    setTimeout(() => { setUploading(false); setUploadDone(true) }, 2200)
  }

  return (
    <div className="min-h-screen pt-14">

      {/* ── HERO ── */}
      <section className="relative overflow-hidden torn-edge"
        style={{ background: 'linear-gradient(160deg, #F0E5CC 0%, #E8D8B0 60%, #F0E5CC 100%)', paddingBottom: 64 }}
      >
        <GreekLetterBg />

        {/* Decorative top-left corner ornament */}
        <div className="absolute top-8 left-8 w-20 h-20 opacity-20 pointer-events-none" aria-hidden>
          <svg viewBox="0 0 80 80" fill="none">
            <path d="M4 4 L4 40 M4 4 L40 4" stroke="#8B6914" strokeWidth="1.5" />
            <path d="M4 14 L14 4" stroke="#8B6914" strokeWidth="0.8" />
            <circle cx="4" cy="4" r="3" fill="#8B6914" />
          </svg>
        </div>
        <div className="absolute top-8 right-8 w-20 h-20 opacity-20 pointer-events-none rotate-90" aria-hidden>
          <svg viewBox="0 0 80 80" fill="none">
            <path d="M4 4 L4 40 M4 4 L40 4" stroke="#8B6914" strokeWidth="1.5" />
            <circle cx="4" cy="4" r="3" fill="#8B6914" />
          </svg>
        </div>

        <div className="max-w-6xl mx-auto px-6 pt-16 pb-8 flex flex-col lg:flex-row items-center gap-12">

          {/* Left: text */}
          <div className="flex-1 min-w-0">
            <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }}>
              {/* Stamp badge */}
              <div className="inline-flex items-center gap-2 mb-6">
                <div className="stamp w-10 h-10 flex items-center justify-center text-xs font-mono font-bold"
                  style={{ color: 'var(--accent)', borderColor: 'var(--accent)' }}>
                  79<br/>AD
                </div>
                <span className="text-xs font-mono uppercase tracking-widest" style={{ color: 'var(--text-muted)' }}>
                  PHercParis4 · Scroll 1 · Herculaneum
                </span>
              </div>

              <h1 className="font-serif text-5xl lg:text-6xl font-bold leading-tight mb-4"
                style={{ color: 'var(--text)' }}>
                Read the<br />
                <em className="not-italic" style={{ color: 'var(--accent)' }}>Unreadable.</em>
              </h1>
              <p className="text-lg leading-relaxed mb-8" style={{ color: 'var(--text-mid)', maxWidth: 480 }}>
                A 2,000-year-old papyrus scroll, carbonized by Vesuvius. Upload CT scan data and watch a
                neural network decipher ancient Greek — letter by letter.
              </p>

              <div className="flex flex-wrap gap-3">
                <button
                  onClick={() => document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' })}
                  className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm transition-all active:scale-95"
                  style={{ background: 'var(--accent)', color: '#fff', boxShadow: '0 4px 16px rgba(139,26,26,0.35)' }}
                >
                  <Upload size={16} /> Upload Scroll Data
                </button>
                <button
                  onClick={() => document.getElementById('segments-section')?.scrollIntoView({ behavior: 'smooth' })}
                  className="flex items-center gap-2 px-6 py-3 rounded-xl font-semibold text-sm border transition-all"
                  style={{ color: 'var(--text-mid)', borderColor: 'var(--border)', background: 'rgba(255,255,255,0.4)' }}
                >
                  Browse Segments <ArrowRight size={15} />
                </button>
              </div>

              {/* Stat bar */}
              <div className="flex gap-6 mt-10">
                {[
                  { n: '11', label: 'Segments' },
                  { n: '72 GB', label: 'CT Data' },
                  { n: '33', label: 'Z-layers' },
                  { n: '0.24M', label: 'Params' },
                ].map(({ n, label }) => (
                  <div key={label} className="text-center">
                    <p className="font-serif font-bold text-xl" style={{ color: 'var(--text)' }}>{n}</p>
                    <p className="text-xs" style={{ color: 'var(--text-muted)' }}>{label}</p>
                  </div>
                ))}
              </div>
            </motion.div>
          </div>

          {/* Right: scroll illustration */}
          <motion.div
            initial={{ opacity: 0, x: 30 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: 0.9, delay: 0.2 }}
            className="flex-shrink-0 animate-drift"
          >
            <ScrollSVG className="w-[220px] lg:w-[280px] drop-shadow-2xl" />
          </motion.div>
        </div>
      </section>

      {/* ── HOW IT WORKS ── */}
      <section className="py-20 px-6" style={{ background: 'var(--bg)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-12">
            <p className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: 'var(--text-muted)' }}>The Pipeline</p>
            <h2 className="font-serif text-3xl font-bold" style={{ color: 'var(--text)' }}>From X-rays to Ancient Greek</h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {STEPS.map((step, i) => (
              <motion.div
                key={step.roman}
                initial={{ opacity: 0, y: 16 }} whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }} transition={{ delay: i * 0.1 }}
                className="card-papyrus rounded-2xl p-5 relative overflow-hidden"
              >
                {/* Large Roman numeral watermark */}
                <div className="absolute top-2 right-3 font-serif font-bold text-6xl leading-none pointer-events-none select-none"
                  style={{ color: 'rgba(139,105,20,0.07)' }}>
                  {step.roman}
                </div>
                <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4"
                  style={{ background: 'rgba(139,26,26,0.1)', border: '1px solid rgba(139,26,26,0.2)' }}>
                  <step.icon size={20} style={{ color: 'var(--accent)' }} />
                </div>
                <p className="font-serif font-semibold text-base mb-2" style={{ color: 'var(--text)' }}>{step.title}</p>
                <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>{step.desc}</p>

                {i < STEPS.length - 1 && (
                  <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-10">
                    <ArrowRight size={16} style={{ color: 'var(--border)' }} />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── UPLOAD ── */}
      <section id="upload-section" className="py-20 px-6 relative overflow-hidden"
        style={{ background: 'linear-gradient(180deg, var(--bg) 0%, var(--bg-surface) 100%)' }}>
        <div className="absolute inset-0 pointer-events-none" aria-hidden>
          <div className="absolute top-0 left-0 right-0 h-px" style={{ background: 'var(--border-light)' }} />
        </div>

        <div className="max-w-2xl mx-auto">
          <div className="text-center mb-8">
            <p className="text-xs font-mono uppercase tracking-widest mb-2" style={{ color: 'var(--text-muted)' }}>Unknown Scroll</p>
            <h2 className="font-serif text-3xl font-bold mb-3" style={{ color: 'var(--text)' }}>Upload Your Scroll Data</h2>
            <p className="text-sm leading-relaxed" style={{ color: 'var(--text-muted)' }}>
              Drop your own surface volume TIFFs or NumPy arrays. The pipeline runs locally — no data leaves your machine.
            </p>
          </div>

          {!uploadDone ? (
            <UploadZone onUpload={handleUpload} />
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
              className="card-papyrus rounded-2xl p-8 text-center"
            >
              {/* Animated stamp */}
              <motion.div
                initial={{ scale: 0, rotate: -15 }} animate={{ scale: 1, rotate: -3 }}
                transition={{ type: 'spring', stiffness: 200, damping: 18, delay: 0.1 }}
                className="inline-flex flex-col items-center justify-center w-24 h-24 rounded-full border-4 mb-4"
                style={{ borderColor: 'var(--olive)', color: 'var(--olive)' }}
              >
                <span className="font-serif font-bold text-2xl leading-none">✓</span>
                <span className="text-xs font-mono mt-1">LOADED</span>
              </motion.div>
              <p className="font-serif text-xl font-semibold mb-2" style={{ color: 'var(--text)' }}>Files received</p>
              <p className="text-sm mb-5" style={{ color: 'var(--text-muted)' }}>Select a sample segment below to see how decipherment works, then run your own data locally.</p>
              <button onClick={() => setUploadDone(false)}
                className="text-sm underline" style={{ color: 'var(--text-muted)' }}>
                Upload different files
              </button>
            </motion.div>
          )}

          {uploading && (
            <div className="mt-4 text-center text-sm" style={{ color: 'var(--text-muted)' }}>
              <motion.span animate={{ opacity: [1, 0.4, 1] }} transition={{ repeat: Infinity, duration: 1.2 }}>
                Processing…
              </motion.span>
            </div>
          )}
        </div>
      </section>

      {/* ── SEGMENTS GRID ── */}
      <section id="segments-section" className="py-20 px-6" style={{ background: 'var(--bg)' }}>
        <div className="max-w-5xl mx-auto">
          <div className="flex items-end justify-between mb-8">
            <div>
              <p className="text-xs font-mono uppercase tracking-widest mb-1" style={{ color: 'var(--text-muted)' }}>Sample Data</p>
              <h2 className="font-serif text-3xl font-bold" style={{ color: 'var(--text)' }}>11 Labelled Segments</h2>
            </div>
            <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>~72 GB · PHercParis4</span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {loading
              ? Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-40 rounded-2xl" />)
              : manifest?.segments.map((seg, i) => (
                <motion.button
                  key={seg.id}
                  initial={{ opacity: 0, y: 10 }} whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true }} transition={{ delay: i * 0.04 }}
                  onClick={() => navigate(`/viewer/${seg.id}`)}
                  className="card-papyrus group text-left rounded-2xl p-5 cursor-pointer transition-all hover:-translate-y-0.5 active:scale-[0.99]"
                  style={{ boxShadow: '0 2px 12px rgba(92,60,20,0.1)' }}
                >
                  <div className="flex items-start justify-between mb-2">
                    <div>
                      <div className="flex items-center gap-2">
                        <p className="font-serif font-semibold text-base" style={{ color: 'var(--text)' }}>{seg.label}</p>
                        {seg.realLabel && (
                          <span className="text-xs px-1.5 py-0.5 rounded font-mono"
                            style={{ background: 'rgba(74,94,58,0.12)', color: 'var(--olive)', border: '1px solid rgba(74,94,58,0.25)' }}>
                            labels ✓
                          </span>
                        )}
                      </div>
                      <p className="text-xs font-mono mt-0.5" style={{ color: 'var(--text-muted)' }}>{seg.id}</p>
                    </div>
                    <ArrowRight size={15} className="group-hover:translate-x-0.5 transition-transform mt-0.5 shrink-0"
                      style={{ color: 'var(--text-muted)' }} />
                  </div>

                  <p className="text-sm leading-snug mb-4 line-clamp-2" style={{ color: 'var(--text-mid)' }}>
                    {seg.description}
                  </p>

                  <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
                    <span className="flex items-center gap-1"><Database size={11} /> {formatBytes(seg.sizeMb)}</span>
                    <span className="flex items-center gap-1"><Layers size={11} /> {seg.layers} Z</span>
                    <span>{seg.letters.length} letters</span>
                  </div>

                  {/* Bottom accent bar */}
                  <div className="mt-4 h-0.5 rounded-full"
                    style={{
                      background: `linear-gradient(90deg, var(--accent) 0%, transparent ${Math.min(100, (seg.letters.length / 8) * 100)}%)`,
                      opacity: 0.35,
                    }}
                  />
                </motion.button>
              ))}
          </div>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="py-10 px-6 text-center border-t" style={{ borderColor: 'var(--border-light)' }}>
        <div className="font-serif text-2xl tracking-widest mb-2 opacity-20" style={{ color: 'var(--text)' }}>
          ΑΒΓΔΕΖΗΘΙΚ
        </div>
        <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
          Built for the{' '}
          <a href="https://scrollprize.org" target="_blank" rel="noreferrer"
            className="underline underline-offset-2" style={{ color: 'var(--text-mid)' }}>
            Vesuvius Challenge
          </a>
          {' '} · PHercParis4 · 79 AD
        </p>
      </footer>
    </div>
  )
}
