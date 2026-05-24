import { useCallback, useEffect, useRef, useState } from 'react'
import type { AnimationPhase } from '@/types/segment'

const PHASE_DURATIONS: Record<AnimationPhase, number> = {
  idle: 0,
  ct: 2200,
  ink: 2800,
  letters: 3500,
  done: 0,
}

const ORDERED_PHASES: AnimationPhase[] = ['ct', 'ink', 'letters', 'done']

export function useAnimationSequence(autoPlay = false) {
  const [phase, setPhase] = useState<AnimationPhase>('idle')
  const timersRef = useRef<ReturnType<typeof setTimeout>[]>([])

  const clearTimers = useCallback(() => {
    timersRef.current.forEach(timer => clearTimeout(timer))
    timersRef.current = []
  }, [])

  const start = useCallback(() => {
    clearTimers()
    let elapsed = 0
    ORDERED_PHASES.forEach(nextPhase => {
      timersRef.current.push(setTimeout(() => setPhase(nextPhase), elapsed))
      elapsed += PHASE_DURATIONS[nextPhase]
    })
  }, [clearTimers])

  const reset = useCallback(() => {
    clearTimers()
    setPhase('idle')
  }, [clearTimers])

  useEffect(() => {
    if (autoPlay) timersRef.current.push(setTimeout(start, 0))
    return clearTimers
  }, [autoPlay, clearTimers, start])

  return { phase, start, reset }
}
