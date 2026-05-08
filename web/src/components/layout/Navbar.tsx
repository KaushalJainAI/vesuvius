import { Link, useLocation } from 'react-router-dom'
import { ScrollText } from 'lucide-react'

export function Navbar() {
  const { pathname } = useLocation()
  const onViewer = pathname.startsWith('/viewer')

  return (
    <nav
      className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-3"
      style={{
        background: 'rgba(245,237,216,0.88)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(200,184,138,0.5)',
        boxShadow: '0 1px 12px rgba(92,60,20,0.08)',
      }}
    >
      <Link to="/" className="flex items-center gap-2.5 group">
        <div
          className="w-8 h-8 rounded-lg flex items-center justify-center"
          style={{ background: 'var(--accent)', boxShadow: '0 2px 8px rgba(139,26,26,0.3)' }}
        >
          <ScrollText size={16} color="#fff" />
        </div>
        <div className="leading-tight">
          <span className="font-serif font-semibold text-base" style={{ color: 'var(--text)' }}>Vesuvius</span>
          <span className="hidden sm:inline text-xs ml-1.5 font-mono px-1.5 py-0.5 rounded"
            style={{ color: 'var(--text-muted)', background: 'rgba(139,105,20,0.08)' }}>
            PHercParis4
          </span>
        </div>
      </Link>

      <div className="flex items-center gap-1">
        <Link
          to="/"
          className="px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all"
          style={{
            color: onViewer ? 'var(--text-muted)' : 'var(--accent)',
            background: onViewer ? 'transparent' : 'rgba(139,26,26,0.08)',
          }}
        >
          Segments
        </Link>
        <a
          href="https://scrollprize.org"
          target="_blank"
          rel="noreferrer"
          className="px-3.5 py-1.5 rounded-lg text-sm font-medium transition-all"
          style={{ color: 'var(--text-muted)' }}
        >
          Challenge ↗
        </a>
      </div>
    </nav>
  )
}
