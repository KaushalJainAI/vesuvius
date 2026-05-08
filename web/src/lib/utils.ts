export function cn(...classes: (string | undefined | false | null)[]): string {
  return classes.filter(Boolean).join(' ')
}

export function formatBytes(mb: number): string {
  if (mb >= 1000) return `${(mb / 1000).toFixed(1)} GB`
  return `${mb} MB`
}

export function confidenceColor(c: number): string {
  if (c >= 0.8) return '#4ade80'
  if (c >= 0.6) return '#facc15'
  return '#f87171'
}
