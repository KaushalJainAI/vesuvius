export interface LetterBBox {
  x: number
  y: number
  w: number
  h: number
  char: string
  confidence: number
}

export interface SegmentMeta {
  id: string
  label: string
  sizeMb: number
  width: number
  height: number
  layers: number
  description: string
  ctPreview: string
  inkProb: string
  realLabel?: string
  letters: LetterBBox[]
}

export interface Manifest {
  segments: SegmentMeta[]
}

export type AnimationPhase = 'idle' | 'ct' | 'ink' | 'letters' | 'done'
