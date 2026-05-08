import { useCallback, useEffect, useRef, useState } from 'react'
import type { AnimationPhase } from '@/types/segment'

const PHASE_DURATIONS: Record<AnimationPhase, number> = {
  idle: 0,
  ct: 2200,
  ink: 2800,
  letters: 3500,
  done: 0,
}

export function useAnimationSequence(autoPlay = false) {
  const [phase, setPhase] = useState<AnimationPhase>('idle')
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const advance = useCallback((from: AnimationPhase) => {
    const next: Record<AnimationPhase, AnimationPhase> = {
      idle: 'ct',
      ct: 'ink',
      ink: 'letters',
      letters: 'done',
      done: 'done',
    }
    const nextPhase = next[from]
    setPhase(nextPhase)
    if (nextPhase !== 'done' && PHASE_DURATIONS[nextPhase] > 0) {
      timerRef.current = setTimeout(() => advance(nextPhase), PHASE_DURATIONS[nextPhase])
    }
  }, [])

  const start = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setPhase('ct')
    timerRef.current = setTimeout(() => advance('ct'), PHASE_DURATIONS.ct)
  }, [advance])

  const reset = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    setPhase('idle')
  }, [])

  useEffect(() => {
    if (autoPlay) start()
    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [autoPlay, start])

  return { phase, start, reset }
}
