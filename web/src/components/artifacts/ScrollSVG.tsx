export function ScrollSVG({ className = '' }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 320 480"
      className={className}
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
    >
      {/* Shadow */}
      <ellipse cx="160" cy="455" rx="90" ry="14" fill="rgba(92,60,20,0.18)" />

      {/* Main scroll body */}
      <rect x="40" y="50" width="240" height="360" rx="8" fill="#2A1608" />

      {/* Worn surface grain lines */}
      {[80,110,140,170,200,230,260,290,320,350,380].map((y, i) => (
        <line key={i} x1="48" y1={y} x2="272" y2={y + (i%3-1)*3} stroke="rgba(255,220,160,0.07)" strokeWidth="1" />
      ))}

      {/* Top roller */}
      <rect x="25" y="36" width="270" height="34" rx="17" fill="#3D1F08" />
      <rect x="30" y="42" width="260" height="22" rx="11" fill="#4A2510" />
      <rect x="38" y="46" width="244" height="14" rx="7" fill="#5C3018" />
      {/* Top roller highlight */}
      <rect x="40" y="47" width="240" height="4" rx="2" fill="rgba(255,200,120,0.15)" />

      {/* Bottom roller */}
      <rect x="25" y="390" width="270" height="34" rx="17" fill="#3D1F08" />
      <rect x="30" y="396" width="260" height="22" rx="11" fill="#4A2510" />
      <rect x="38" y="400" width="244" height="14" rx="7" fill="#5C3018" />

      {/* Roller end caps */}
      <circle cx="25"  cy="53" r="12" fill="#6B3820" />
      <circle cx="295" cy="53" r="12" fill="#6B3820" />
      <circle cx="25"  cy="53" r="7"  fill="#8B5030" />
      <circle cx="295" cy="53" r="7"  fill="#8B5030" />
      <circle cx="25"  cy="407" r="12" fill="#6B3820" />
      <circle cx="295" cy="407" r="12" fill="#6B3820" />
      <circle cx="25"  cy="407" r="7"  fill="#8B5030" />
      <circle cx="295" cy="407" r="7"  fill="#8B5030" />

      {/* Papyrus text lines (faded) */}
      <g opacity="0.18" stroke="#D4A850" strokeWidth="0.8">
        {[100,115,130,145,160,175,190,205,220,235,250,265,280,295,310,325,340].map((y, i) => (
          <line key={i} x1={56 + (i%3)*4} y1={y} x2={256 - (i%2)*8} y2={y} strokeDasharray={i%4===0?"0":"4 2"} />
        ))}
      </g>

      {/* Greek letters on scroll */}
      <text x="70" y="135" fontFamily="serif" fontSize="11" fill="rgba(212,168,80,0.35)" letterSpacing="8">ΑΒΓΔΕΖΗΘ</text>
      <text x="80" y="165" fontFamily="serif" fontSize="10" fill="rgba(212,168,80,0.25)" letterSpacing="6">ΙΚΛΜΝΞΟΠ</text>
      <text x="70" y="195" fontFamily="serif" fontSize="11" fill="rgba(212,168,80,0.3)"  letterSpacing="7">ΡΣΤΥΦΧΨΩ</text>
      <text x="85" y="225" fontFamily="serif" fontSize="9"  fill="rgba(212,168,80,0.2)"  letterSpacing="5">ΑΝΤΡΩΠΟΣ</text>
      <text x="72" y="255" fontFamily="serif" fontSize="10" fill="rgba(212,168,80,0.28)" letterSpacing="6">ΛΟΓΟΣ ΚΑΙ</text>
      <text x="80" y="285" fontFamily="serif" fontSize="9"  fill="rgba(212,168,80,0.22)" letterSpacing="5">ΦΥΣΙΣ ΤΩΝ</text>
      <text x="70" y="315" fontFamily="serif" fontSize="10" fill="rgba(212,168,80,0.26)" letterSpacing="6">ΠΡΑΓΜΑΤΩΝ</text>

      {/* Edge wax seal hint */}
      <circle cx="270" cy="230" r="20" fill="rgba(139,26,26,0.6)" />
      <circle cx="270" cy="230" r="16" fill="none" stroke="rgba(212,168,80,0.5)" strokeWidth="1" />
      <text x="270" y="235" textAnchor="middle" fontFamily="serif" fontSize="9" fill="rgba(212,168,80,0.8)">ΦΠ</text>
    </svg>
  )
}
