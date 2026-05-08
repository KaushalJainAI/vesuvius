import { useRef, useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Play, RotateCcw, ChevronRight } from 'lucide-react'
import type { SegmentMeta, AnimationPhase } from '@/types/segment'
import { useAnimationSequence } from '@/hooks/useAnimationSequence'
import { confidenceColor } from '@/lib/utils'

const PHASE_LABELS: Record<AnimationPhase, { roman: string; title: string; desc: string } | null> = {
  idle: null,
  ct:      { roman: 'I',   title: 'CT Scan',        desc: '33 X-ray tomographic slices through the papyrus' },
  ink:     { roman: 'II',  title: 'Ink Detection',   desc: '3D CNN produces per-pixel ink probability map' },
  letters: { roman: 'III', title: 'Letter Isolation', desc: 'Connected components yield Greek letter candidates' },
  done:    { roman: 'IV',  title: 'Transcription',   desc: 'Characters ranked by model confidence score' },
}

interface DecipherAnimationProps {
  segment: SegmentMeta
}

export function DecipherAnimation({ segment }: DecipherAnimationProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [dims, setDims] = useState({ w: 512, h: 384 })
  const { phase, start, reset } = useAnimationSequence()

  useEffect(() => {
    const el = containerRef.current; if (!el) return
    const obs = new ResizeObserver(() => setDims({ w: el.clientWidth, h: el.clientHeight }))
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  useEffect(() => { reset() }, [segment.id, reset])

  const phaseIdx = ['idle','ct','ink','letters','done'].indexOf(phase)
  const showCt      = phaseIdx >= 1
  const showInk     = phaseIdx >= 2
  const showLetters = phaseIdx >= 3
  const showDone    = phase === 'done'

  const scaleX = dims.w / 512
  const scaleY = dims.h / 512

  const label = phase !== 'idle' ? PHASE_LABELS[phase] : null

  return (
    <div className="flex flex-col gap-5">
      {/* Main canvas */}
      <div
        ref={containerRef}
        className="relative w-full rounded-2xl overflow-hidden"
        style={{ minHeight: 380, background: '#0e0905', border: '1.5px solid rgba(200,184,138,0.2)', boxShadow: '0 8px 32px rgba(92,60,20,0.18)' }}
      >
        {/* Phase label badge */}
        <AnimatePresence mode="wait">
          {label && (
            <motion.div
              key={phase}
              initial={{ opacity: 0, y: -6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
              className="absolute top-4 left-4 z-20 flex items-center gap-3"
            >
              <div
                className="w-9 h-9 rounded-full flex items-center justify-center font-serif font-bold text-sm shrink-0"
                style={{ background: 'var(--accent)', color: '#fff', boxShadow: '0 0 12px rgba(139,26,26,0.5)' }}
              >
                {label.roman}
              </div>
              <div className="rounded-xl px-3 py-1.5" style={{ background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(8px)' }}>
                <p className="text-xs font-semibold" style={{ color: 'var(--gold-light)' }}>{label.title}</p>
                <p className="text-xs opacity-60 text-white">{label.desc}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* IDLE state */}
        {phase === 'idle' && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3">
            <div className="font-serif text-5xl tracking-widest" style={{ color: 'rgba(212,168,80,0.2)' }}>ΑΒΓΔΕΖ</div>
            <p className="text-sm" style={{ color: 'rgba(255,255,255,0.25)' }}>Press <strong style={{ color: 'rgba(255,255,255,0.45)' }}>Play</strong> to begin decipherment</p>
          </div>
        )}

        {/* CT layer */}
        <motion.div
          className="absolute inset-0"
          initial={{ opacity: 0 }} animate={{ opacity: showCt ? 1 : 0 }} transition={{ duration: 1.0 }}
        >
          <img src={segment.ctPreview} alt="CT slice" className="w-full h-full object-cover opacity-80"
            onError={e => (e.target as HTMLImageElement).style.opacity = '0'} />
          {/* Procedural papyrus fallback */}
          <div className="absolute inset-0" style={{
            background: 'radial-gradient(ellipse 80% 70% at 50% 50%, #2a1a08 0%, #0e0905 100%)',
            mixBlendMode: 'multiply',
          }} />
          {/* CT scan-line sweep effect */}
          {showCt && !showInk && (
            <div className="absolute inset-0 overflow-hidden pointer-events-none">
              <motion.div
                initial={{ top: '-5%' }} animate={{ top: '105%' }}
                transition={{ duration: 2.0, ease: 'linear' }}
                className="absolute left-0 right-0 h-1"
                style={{ background: 'linear-gradient(0deg, transparent, rgba(100,200,255,0.35), transparent)' }}
              />
            </div>
          )}
        </motion.div>

        {/* INK probability overlay */}
        <motion.div
          className="absolute inset-0 pointer-events-none"
          initial={{ opacity: 0 }} animate={{ opacity: showInk ? 1 : 0 }} transition={{ duration: 1.4, ease: 'easeInOut' }}
        >
          <img src={segment.inkProb} alt="Ink probability" className="w-full h-full object-cover"
            style={{ mixBlendMode: 'screen' }}
            onError={e => (e.target as HTMLImageElement).style.opacity = '0'} />
          {/* Synthetic heatmap glow when image missing */}
          <div className="absolute inset-0" style={{
            background: 'radial-gradient(ellipse 55% 35% at 42% 48%, rgba(255,140,30,0.38) 0%, rgba(255,60,10,0.18) 45%, transparent 75%)',
            mixBlendMode: 'screen',
          }} />
        </motion.div>

        {/* LETTER bounding boxes */}
        {segment.letters.map((l, i) => {
          const x = l.x * scaleX, y = l.y * scaleY
          const w = l.w * scaleX, h = l.h * scaleY
          const col = confidenceColor(l.confidence)
          return (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0.4 }}
              animate={showLetters ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.4 }}
              transition={{ delay: showLetters ? i * 0.13 : 0, duration: 0.38, type: 'spring', stiffness: 300, damping: 20 }}
              className="absolute pointer-events-none"
              style={{ left: x, top: y, width: w, height: h }}
            >
              <div className="absolute inset-0 rounded-sm"
                style={{ border: `1.5px solid ${col}`, boxShadow: `0 0 8px ${col}50` }} />
              <div className="absolute -top-5 left-0 font-mono font-bold text-xs px-1.5 rounded"
                style={{ color: col, background: `${col}20`, fontSize: 11, lineHeight: '1.4' }}>
                {l.char}
              </div>
            </motion.div>
          )
        })}

        {/* Real label image reveal overlay */}
        <AnimatePresence>
          {showDone && segment.realLabel && (
            <motion.div
              initial={{ opacity: 0, clipPath: 'inset(0 100% 0 0)' }}
              animate={{ opacity: 1, clipPath: 'inset(0 0% 0 0)' }}
              transition={{ duration: 1.5, ease: [0.4, 0, 0.2, 1] }}
              className="absolute inset-0 z-10"
            >
              <img src={segment.realLabel} alt="Ink labels" className="w-full h-full object-contain"
                style={{ background: '#0e0905' }} />
              <div className="absolute top-4 right-4 px-2.5 py-1 rounded-lg text-xs font-mono"
                style={{ background: 'rgba(0,0,0,0.7)', color: 'var(--gold-light)' }}>
                ink_labels.tif
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Controls bar */}
      <div className="flex items-center justify-between">
        <button
          onClick={phase === 'idle' || phase === 'done' ? start : reset}
          className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all active:scale-95"
          style={{ background: 'var(--accent)', color: '#fff', boxShadow: '0 2px 10px rgba(139,26,26,0.3)' }}
        >
          {phase === 'idle' || phase === 'done'
            ? <><Play size={14} fill="#fff" /> Play Sequence</>
            : <><RotateCcw size={14} /> Reset</>}
        </button>

        {/* Phase progress bar */}
        <div className="flex items-center gap-2">
          {(['ct','ink','letters','done'] as AnimationPhase[]).map((p, i) => (
            <div key={p} className="flex items-center gap-2">
              <div
                className="h-1.5 rounded-full transition-all duration-700"
                style={{
                  width: phaseIdx > i+1 ? 32 : phaseIdx === i+1 ? 24 : 16,
                  background: phaseIdx > i+1 ? 'var(--accent)' : phaseIdx === i+1 ? 'rgba(139,26,26,0.5)' : 'rgba(139,105,20,0.2)',
                }}
              />
            </div>
          ))}
          <span className="text-xs ml-1 font-mono" style={{ color: 'var(--text-muted)' }}>
            {phaseIdx}/4
          </span>
        </div>
      </div>

      {/* Transcription result */}
      <AnimatePresence>
        {showDone && (
          <motion.div
            initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
            className="rounded-2xl p-5"
            style={{ background: 'var(--bg-elevated)', border: '1px solid var(--border-light)', boxShadow: '0 4px 16px rgba(92,60,20,0.1)' }}
          >
            <p className="text-xs font-mono uppercase tracking-widest mb-3" style={{ color: 'var(--text-muted)' }}>
              Decoded text · {segment.letters.length} characters
            </p>
            <p className="font-serif text-2xl tracking-widest mb-4" style={{ color: 'var(--text)', letterSpacing: '0.15em' }}>
              {segment.letters.map(l => l.char).join(' ')}
            </p>
            <div className="flex flex-wrap gap-2">
              {segment.letters.map((l, i) => {
                const col = confidenceColor(l.confidence)
                return (
                  <motion.span
                    key={i}
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.06 }}
                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-sm font-mono border"
                    style={{ borderColor: `${col}40`, color: col, background: `${col}10` }}
                  >
                    {l.char}
                    <span style={{ opacity: 0.55, fontSize: 10 }}>{(l.confidence*100).toFixed(0)}%</span>
                  </motion.span>
                )
              })}
            </div>
            {!segment.realLabel && (
              <p className="mt-4 flex items-center gap-1.5 text-xs" style={{ color: 'var(--text-muted)' }}>
                <ChevronRight size={11} />
                Placeholder labels — run <code className="font-mono px-1.5 py-0.5 rounded text-xs" style={{ background: 'rgba(139,105,20,0.1)' }}>scripts/build_web_assets.py</code> for real predictions
              </p>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
