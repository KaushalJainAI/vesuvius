import { useEffect, useState } from 'react'
import { Info, ScrollText, Sparkles } from 'lucide-react'

interface Line {
  line_no: number
  y_band: [number, number]
  x_range: [number, number]
  blob_bboxes: number[][]
  n_blobs: number
}

interface LabelImage {
  path: string
  width: number
  height: number
  native_width: number
  native_height: number
}

interface SegmentResult {
  seg_id: string
  created: string
  scroll?: string
  source: string
  label_image: LabelImage | null
  transcription_available: boolean
  note?: string
  n_lines: number
  n_blobs: number
  lines: Line[]
}

interface TranscriptionResult {
  n_strips?: number
  segment_text?: string
  segment_meaning?: string
  segment_translation_en?: string
  probable_scroll_summary?: string
  segment_summary?: {
    text?: string
    probable_summary?: string
    english_translation?: string
    probable_scroll_summary?: string
    confidence?: string
  }
  alignment_summary?: {
    n_character_boxes?: number
    n_component_snapped?: number
    n_linked_to_blobs?: number
  }
  character_boxes?: CharacterBox[]
  strips?: Array<{
    strip_id: number
    consensus?: {
      text?: string
      translation_en?: string
    }
  }>
}

interface CharacterBox {
  strip_id?: number
  char_index?: number
  char?: string
  confidence?: number
  tier?: 'HIGH' | 'MED' | 'LOW'
  bbox_full?: [number, number, number, number]
  line_no?: number | null
  linked_blob_indices?: number[]
  alignment_source?: string
}

interface Props {
  segmentId: string
}

export function DecipherPanel({ segmentId }: Props) {
  const [data, setData] = useState<SegmentResult | null>(null)
  const [transcription, setTranscription] = useState<TranscriptionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [hovered, setHovered] = useState<{ line: number; idx: number } | null>(null)

  useEffect(() => {
    let cancelled = false

    const detectionRequest = fetch(`/assets/decipher/${segmentId}/result.detection.json`)
      .then(r => r.ok ? r : fetch(`/assets/decipher/${segmentId}/result.json`))
      .then(r => {
        if (!r.ok) throw new Error(`no detection result for ${segmentId}`)
        return r.json() as Promise<SegmentResult>
      })

    const transcriptionRequest = fetch(`/assets/decipher/${segmentId}/result.json`)
      .then(r => r.ok ? r.json() as Promise<TranscriptionResult> : null)
      .then(d => {
        if (!d || !Array.isArray(d.strips) || d.strips.length === 0) return null
        return d
      })
      .catch(() => null)

    Promise.all([detectionRequest, transcriptionRequest])
      .then(([d, t]) => {
        if (cancelled) return
        setError(null)
        setData(d)
        setTranscription(t)
      })
      .catch(e => {
        if (cancelled) return
        setData(null)
        setTranscription(null)
        setError(String(e))
      })

    return () => { cancelled = true }
  }, [segmentId])

  if (error) {
    return (
      <div className="rounded-2xl p-6 card-papyrus">
        <div className="flex items-start gap-3">
          <Info size={18} style={{ color: 'var(--text-muted)' }} className="mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-serif font-semibold mb-1" style={{ color: 'var(--text)' }}>
              No detection data available yet
            </p>
            <p className="text-sm mb-2" style={{ color: 'var(--text-muted)' }}>
              Label-based detection has not been built for this segment.
            </p>
            <p className="text-xs font-mono p-2 rounded"
               style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-mid)' }}>
              python scripts/build_all_decipher.py
            </p>
          </div>
        </div>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="rounded-2xl p-6 card-papyrus">
        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading detection...</p>
      </div>
    )
  }

  const lines = Array.isArray(data.lines) ? data.lines : []
  const segmentText = transcription?.segment_summary?.text || transcription?.segment_text || ''
  const segmentMeaning = transcription?.segment_summary?.probable_summary || transcription?.segment_meaning || ''
  const segmentTranslation = transcription?.segment_summary?.english_translation || transcription?.segment_translation_en || ''
  const scrollSummary = transcription?.segment_summary?.probable_scroll_summary || transcription?.probable_scroll_summary || segmentMeaning
  const rawSegmentConfidence = transcription?.segment_summary?.confidence || 'provisional'
  const segmentConfidence = rawSegmentConfidence === 'medium' ? 'provisional' : rawSegmentConfidence
  const stripTexts = transcription?.strips
    ?.map(s => s.consensus?.text)
    .filter((text): text is string => Boolean(text && text.trim())) ?? []
  const hasTranscription = Boolean(segmentText || stripTexts.length)

  if (!data.label_image || lines.length === 0) {
    return (
      <div className="rounded-2xl p-6 card-papyrus">
        <div className="flex items-start gap-3">
          <Info size={18} style={{ color: 'var(--text-muted)' }} className="mt-0.5 flex-shrink-0" />
          <div>
            <p className="font-serif font-semibold mb-1" style={{ color: 'var(--text)' }}>
              No label detection for this segment
            </p>
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {data.note || 'No enhanced-label detection has been generated yet.'}
            </p>
            <p className="text-xs font-mono mt-2 p-2 rounded"
               style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-mid)' }}>
              python scripts/build_all_decipher.py
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="card-papyrus rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-3 flex-wrap">
          <ScrollText size={18} style={{ color: 'var(--accent)' }} />
          <h3 className="font-serif text-lg font-bold" style={{ color: 'var(--text)' }}>
            Text Deciphering
          </h3>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded uppercase tracking-widest"
                style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-muted)' }}>
            {lines.length} lines · {data.n_blobs} blobs
          </span>
        </div>

        {data.scroll && (
          <p className="text-xs mb-1" style={{ color: 'var(--text-muted)' }}>
            <span className="font-mono">{data.scroll}</span>
          </p>
        )}

        <div className="rounded-lg p-3 mt-2"
             style={{ background: 'rgba(202,161,90,0.08)',
                      border: '1px solid rgba(202,161,90,0.25)' }}>
          <p className="text-[10px] uppercase font-mono tracking-widest mb-1.5 flex items-center gap-1.5"
             style={{ color: 'var(--text-muted)' }}>
            <Sparkles size={11} /> Status
          </p>
          {hasTranscription && (
            <div className="space-y-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded uppercase tracking-widest"
                      style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-muted)' }}>
                  {transcription?.n_strips ?? stripTexts.length} strips
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded uppercase tracking-widest"
                      style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-muted)' }}>
                  {segmentConfidence}
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="rounded-md p-3"
                     style={{ background: 'rgba(92,60,20,0.045)', border: '1px solid var(--border-light)' }}>
                  <p className="text-[10px] uppercase font-mono tracking-widest mb-1"
                     style={{ color: 'var(--text-muted)' }}>English translation</p>
                  <p className="font-serif text-sm leading-relaxed" style={{ color: 'var(--text)' }}>
                    {segmentTranslation && segmentTranslation !== '[uncertain]' ? segmentTranslation : 'No stable English translation yet.'}
                  </p>
                </div>
                <div className="rounded-md p-3"
                     style={{ background: 'rgba(92,60,20,0.045)', border: '1px solid var(--border-light)' }}>
                  <p className="text-[10px] uppercase font-mono tracking-widest mb-1"
                     style={{ color: 'var(--text-muted)' }}>Probable scroll summary</p>
                  <p className="font-serif text-sm leading-relaxed" style={{ color: 'var(--text)' }}>
                    {scrollSummary || 'No stable scroll-level summary yet.'}
                  </p>
                </div>
              </div>
              <pre className="font-serif text-base leading-relaxed whitespace-pre-wrap break-words p-3 rounded-md"
                   style={{
                     color: 'var(--text-mid)',
                     background: 'rgba(92,60,20,0.06)',
                     border: '1px solid var(--border-light)',
                     letterSpacing: '0.08em',
                   }}>{segmentText || stripTexts.join('\n')}</pre>
            </div>
          )}
          <p className="font-serif text-sm leading-relaxed" style={{ color: 'var(--text)', display: hasTranscription ? 'none' : undefined }}>
            Transcription has not been run on this segment. What you see below is
            the enhanced ink-label image with one bounding box per detected
            connected-component ink blob - no Greek characters, transliterations,
            or translations have been assigned. Hover a blob to highlight it.
          </p>
        </div>
      </div>

      {/* Aligned label + per-line blob counts */}
      <AlignedDetection
        segmentId={segmentId}
        labelImage={data.label_image}
        lines={lines}
        characterBoxes={transcription?.character_boxes ?? []}
        hovered={hovered}
        setHovered={setHovered}
      />
    </div>
  )
}

interface AlignedProps {
  segmentId: string
  labelImage: LabelImage
  lines: Line[]
  characterBoxes: CharacterBox[]
  hovered: { line: number; idx: number } | null
  setHovered: (h: { line: number; idx: number } | null) => void
}

function AlignedDetection({
  segmentId, labelImage, lines, characterBoxes, hovered, setHovered,
}: AlignedProps) {
  const { path, width, height, native_width, native_height } = labelImage
  const src = `/assets/decipher/${segmentId}/${path}`
  const charsByLine = characterBoxes.reduce<Record<number, CharacterBox[]>>((acc, box) => {
    if (box.line_no == null || box.line_no < 0) return acc
    if (!acc[box.line_no]) acc[box.line_no] = []
    acc[box.line_no].push(box)
    return acc
  }, {})

  // SVG viewBox uses the rendered label-image pixel space so coordinates
  // scale with the container instead of overflowing into the viewer width.
  return (
    <div className="card-papyrus rounded-2xl">
      <div className="px-5 py-3 flex items-center justify-between flex-wrap gap-2 rounded-t-2xl"
           style={{ background: 'rgba(92,60,20,0.04)',
                    borderBottom: '1px solid var(--border-light)' }}>
        <div className="flex items-center gap-2">
          <span className="font-serif font-semibold text-sm"
                style={{ color: 'var(--text)' }}>
            Text deciphering overlay
          </span>
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                style={{ background: 'rgba(92,60,20,0.08)', color: 'var(--text-muted)' }}>
            hover a box to highlight its line
          </span>
        </div>
      </div>

      <div className="grid gap-4 p-4 grid-cols-1 md:grid-cols-[minmax(0,1fr)_minmax(180px,260px)]">
        {/* LEFT: rendered label + svg overlay (responsive) */}
        <div className="self-start md:sticky md:top-32 w-full"
             style={{ background: '#0a0a09', borderRadius: 8, overflow: 'hidden' }}>
          <div className="relative w-full" style={{ aspectRatio: `${width} / ${height}` }}>
            <img src={src} alt="enhanced label - full segment"
                 className="absolute inset-0 w-full h-full block select-none pointer-events-none"
                 style={{ imageRendering: 'pixelated', objectFit: 'contain' }} />
            <svg
              className="absolute inset-0 w-full h-full"
              viewBox={`0 0 ${native_width} ${native_height}`}
              preserveAspectRatio="xMidYMid meet"
              style={{ pointerEvents: 'none' }}
            >
              {lines.map(line => line.blob_bboxes.map((b, i) => {
                const [bx, by, bw, bh] = b
                const isHovered = hovered?.line === line.line_no && hovered?.idx === i
                return (
                  <rect
                    key={`${line.line_no}-${i}`}
                    x={bx} y={by} width={bw} height={bh}
                    fill={isHovered ? 'rgba(202,161,90,0.35)' : 'transparent'}
                    stroke={isHovered ? '#caa15a' : 'rgba(202,161,90,0.05)'}
                    strokeWidth={isHovered ? 2 : 1}
                    vectorEffect="non-scaling-stroke"
                    style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                    onMouseEnter={() => setHovered({ line: line.line_no, idx: i })}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <title>{`L${line.line_no} · blob ${i + 1}`}</title>
                  </rect>
                )
              }))}
              {characterBoxes.map((box, i) => {
                if (!box.bbox_full) return null
                const [x, y, w, h] = box.bbox_full
                const lineNo = box.line_no ?? -1
                const tier = box.tier ?? 'LOW'
                const color = tier === 'HIGH' ? '#3fcf6c' : tier === 'MED' ? '#e9a73b' : '#e35a5a'
                return (
                  <g
                    key={`char-${box.strip_id}-${box.char_index}-${i}`}
                    style={{ pointerEvents: 'auto', cursor: 'pointer' }}
                    onMouseEnter={() => setHovered({ line: lineNo, idx: -1000 - i })}
                    onMouseLeave={() => setHovered(null)}
                  >
                    <rect
                      x={x} y={y} width={w} height={h}
                      fill={`${color}10`}
                      stroke={color}
                      strokeWidth={1}
                      vectorEffect="non-scaling-stroke"
                    />
                    <text
                      x={x + w / 2}
                      y={Math.max(12, y - 5)}
                      textAnchor="middle"
                      fontSize="18"
                      fontWeight="700"
                      fill={color}
                      stroke="#0a0a09"
                      strokeWidth="3"
                      paintOrder="stroke"
                      style={{ pointerEvents: 'none' }}
                    >
                      {box.char}
                    </text>
                    <title>{`${box.char ?? '?'} - ${Math.round((box.confidence ?? 0) * 100)}% - ${box.alignment_source ?? 'aligned'}`}</title>
                  </g>
                )
              })}
            </svg>
          </div>
        </div>

        {/* RIGHT: per-line stats */}
        <div className="flex flex-col gap-1.5 min-w-0 max-h-[70vh] overflow-y-auto pr-1">
          {lines.map(line => {
            const isLineHovered = hovered?.line === line.line_no
            const lineChars = charsByLine[line.line_no] ?? []
            const lineText = lineChars.map(c => c.char || '').join('')
            return (
              <div
                key={line.line_no}
                className="px-3 py-1.5 rounded-md transition-colors scroll-mt-20"
                style={{
                  background: isLineHovered
                    ? 'rgba(202,161,90,0.10)'
                    : 'transparent',
                  borderLeft: `2px solid ${
                    isLineHovered ? '#caa15a' : 'rgba(92,60,20,0.15)'
                  }`,
                }}
              >
                <div className="flex items-baseline gap-2 mb-0.5">
                  <span className="text-[10px] font-mono"
                        style={{ color: 'var(--text-muted)' }}>
                    L{line.line_no}
                  </span>
                  <span className="text-[10px] font-mono"
                        style={{ color: 'var(--text-muted)' }}>
                    {line.n_blobs} blob{line.n_blobs === 1 ? '' : 's'}
                  </span>
                  <span className="text-[10px] font-mono"
                        style={{ color: 'var(--text-muted)' }}>
                    y {line.y_band[0]}-{line.y_band[1]}
                  </span>
                </div>
                {lineText ? (
                  <p className="font-serif text-sm break-words"
                     style={{ color: 'var(--text)', letterSpacing: '0.08em' }}>
                    {lineText}
                    <span className="ml-2 text-[10px] font-mono tracking-normal"
                          style={{ color: 'var(--text-muted)' }}>
                      {lineChars.length} mapped
                    </span>
                  </p>
                ) : (
                  <p className="text-[11px] font-mono"
                     style={{ color: 'var(--text-muted)' }}>
                    no mapped letters
                  </p>
                )}
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}


