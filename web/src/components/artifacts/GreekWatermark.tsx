export function GreekWatermark({ className = '' }: { className?: string }) {
  return (
    <div
      className={`greek-watermark select-none pointer-events-none font-serif ${className}`}
      aria-hidden
    >
      ΑΒΓΔΕΖΗ<br />ΘΙΚΛΜΝΞ<br />ΟΠΡΣΤΥΦ<br />ΧΨΩ
    </div>
  )
}

export function GreekLetterBg() {
  return (
    <div
      className="absolute inset-0 overflow-hidden pointer-events-none select-none"
      aria-hidden
    >
      <div
        className="absolute -top-8 -right-12 text-[22rem] font-serif leading-none opacity-[0.03] text-stone-600 rotate-12"
      >
        Α
      </div>
      <div
        className="absolute bottom-0 -left-16 text-[18rem] font-serif leading-none opacity-[0.03] text-stone-600 -rotate-6"
      >
        Ω
      </div>
    </div>
  )
}
