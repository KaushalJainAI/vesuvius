import { cn } from '@/lib/utils'

interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'accent' | 'muted'
  className?: string
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium',
        variant === 'default' && 'bg-white/10 text-white/70',
        variant === 'accent' && 'bg-amber-500/20 text-amber-300 border border-amber-500/30',
        variant === 'muted' && 'bg-white/5 text-white/40',
        className
      )}
    >
      {children}
    </span>
  )
}
