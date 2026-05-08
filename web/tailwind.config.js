/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        serif: ['"Playfair Display"', 'Georgia', 'serif'],
        sans: ['"Source Sans 3"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
        greek: ['"GFS Didot"', '"Playfair Display"', 'serif'],
      },
      colors: {
        parchment: {
          50:  '#fdfaf3',
          100: '#f8f0da',
          200: '#f0e0b5',
          300: '#e4ca88',
          400: '#d4ad58',
          500: '#c09030',
        },
        pompeii: {
          DEFAULT: '#8B1A1A',
          light:   '#b02828',
          dark:    '#5c1010',
        },
        stone: {
          50:  '#f5f1e8',
          100: '#e8e0cc',
          200: '#d4c8a8',
          300: '#b8a880',
          400: '#8c7a52',
          500: '#5c4e30',
          600: '#3a2e18',
          700: '#1c150a',
          800: '#0e0905',
        },
        gold: '#B8860B',
        olive: '#4A5E3A',
      },
      backgroundImage: {
        'papyrus': "repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(139,108,64,0.06) 3px, rgba(139,108,64,0.06) 4px), repeating-linear-gradient(90deg, transparent, transparent 7px, rgba(139,108,64,0.04) 7px, rgba(139,108,64,0.04) 8px)",
        'grain': "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E\")",
      },
      keyframes: {
        'unroll':    { from: { clipPath: 'inset(0 100% 0 0)' }, to: { clipPath: 'inset(0 0% 0 0)' } },
        'scanline':  { from: { transform: 'translateY(-100%)' }, to: { transform: 'translateY(100%)' } },
        'fade-up':   { from: { opacity: '0', transform: 'translateY(20px)' }, to: { opacity: '1', transform: 'translateY(0)' } },
        'stamp':     { '0%': { transform: 'scale(1.4) rotate(-8deg)', opacity: '0' }, '60%': { transform: 'scale(0.96) rotate(1deg)', opacity: '1' }, '100%': { transform: 'scale(1) rotate(0deg)', opacity: '1' } },
        'drift':     { '0%,100%': { transform: 'translateY(0) rotate(0deg)' }, '33%': { transform: 'translateY(-8px) rotate(0.5deg)' }, '66%': { transform: 'translateY(4px) rotate(-0.3deg)' } },
        'letter-in': { from: { opacity: '0', transform: 'scale(0.5) rotate(-10deg)' }, to: { opacity: '1', transform: 'scale(1) rotate(0deg)' } },
        'shimmer':   { from: { backgroundPosition: '-400px 0' }, to: { backgroundPosition: '400px 0' } },
      },
      animation: {
        'unroll':    'unroll 1.2s cubic-bezier(0.4,0,0.2,1) forwards',
        'scanline':  'scanline 1.8s ease-in-out infinite',
        'fade-up':   'fade-up 0.6s ease-out both',
        'stamp':     'stamp 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards',
        'drift':     'drift 8s ease-in-out infinite',
        'letter-in': 'letter-in 0.4s cubic-bezier(0.34,1.56,0.64,1) both',
        'shimmer':   'shimmer 2s linear infinite',
      },
    },
  },
  plugins: [],
}
