import { useCallback, useRef, useState } from 'react'
import { Upload, FolderOpen, AlertCircle } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'

interface UploadedFile {
  name: string
  size: number
  type: string
}

interface UploadZoneProps {
  onUpload: (files: UploadedFile[]) => void
}

export function UploadZone({ onUpload }: UploadZoneProps) {
  const [dragging, setDragging] = useState(false)
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const accept = (fileList: FileList | null) => {
    if (!fileList || fileList.length === 0) return
    setError(null)
    const accepted: UploadedFile[] = []
    const rejected: string[] = []
    Array.from(fileList).forEach(f => {
      const ext = f.name.split('.').pop()?.toLowerCase()
      if (['tif', 'tiff', 'npy', 'zip', 'tar', 'gz'].includes(ext ?? '')) {
        accepted.push({ name: f.name, size: f.size, type: f.type || ext! })
      } else {
        rejected.push(f.name)
      }
    })
    if (rejected.length) setError(`Unsupported: ${rejected.join(', ')} — accepted: .tif .tiff .npy .zip`)
    if (accepted.length) { setFiles(accepted); onUpload(accepted) }
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault(); setDragging(false)
    accept(e.dataTransfer.files)
  }, [])

  const onDragOver = (e: React.DragEvent) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = () => setDragging(false)

  const formatSize = (b: number) => b > 1e9 ? `${(b/1e9).toFixed(1)} GB` : b > 1e6 ? `${(b/1e6).toFixed(1)} MB` : `${(b/1e3).toFixed(0)} KB`

  return (
    <div className="w-full">
      <motion.div
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        animate={{ borderColor: dragging ? 'rgba(139,26,26,0.7)' : 'rgba(200,184,138,0.6)' }}
        className="relative w-full rounded-2xl border-2 border-dashed cursor-pointer transition-all overflow-hidden"
        style={{ background: dragging ? 'rgba(139,26,26,0.04)' : 'rgba(249,243,229,0.7)' }}
        onClick={() => inputRef.current?.click()}
      >
        {/* Archaeological grid overlay */}
        <div
          className="absolute inset-0 opacity-30 pointer-events-none"
          style={{
            backgroundImage: 'linear-gradient(rgba(139,105,20,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(139,105,20,0.1) 1px, transparent 1px)',
            backgroundSize: '24px 24px',
          }}
        />

        <div className="relative flex flex-col items-center justify-center gap-4 py-14 px-6">
          <motion.div
            animate={{ y: dragging ? -6 : 0 }}
            transition={{ type: 'spring', stiffness: 300, damping: 20 }}
            className="w-16 h-16 rounded-2xl flex items-center justify-center"
            style={{ background: 'rgba(139,26,26,0.1)', border: '1.5px solid rgba(139,26,26,0.2)' }}
          >
            <Upload size={28} style={{ color: 'var(--accent)' }} />
          </motion.div>

          <div className="text-center">
            <p className="font-serif text-xl font-semibold" style={{ color: 'var(--text)' }}>
              {dragging ? 'Release to upload' : 'Drop your scroll data here'}
            </p>
            <p className="mt-1.5 text-sm" style={{ color: 'var(--text-muted)' }}>
              Surface volume TIFFs, NumPy arrays, or ZIP archives
            </p>
            <p className="mt-1 text-xs font-mono" style={{ color: 'var(--text-muted)', opacity: 0.7 }}>
              .tif · .tiff · .npy · .zip · .tar.gz
            </p>
          </div>

          <button
            type="button"
            onClick={e => { e.stopPropagation(); inputRef.current?.click() }}
            className="flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-semibold transition-all"
            style={{ background: 'var(--accent)', color: '#fff', boxShadow: '0 2px 8px rgba(139,26,26,0.3)' }}
          >
            <FolderOpen size={15} />
            Browse files
          </button>

          {/* Decorative corner marks */}
          {['top-3 left-3','top-3 right-3','bottom-3 left-3','bottom-3 right-3'].map((pos, i) => (
            <div key={i} className={`absolute ${pos} w-4 h-4`} style={{ borderColor: 'var(--border)', borderWidth: i<2 ? '1.5px 0 0 1.5px' : '0 1.5px 1.5px 0' }} />
          ))}
        </div>
      </motion.div>

      <input
        ref={inputRef} type="file" multiple accept=".tif,.tiff,.npy,.zip,.tar,.gz"
        className="hidden" onChange={e => accept(e.target.files)}
      />

      <AnimatePresence>
        {error && (
          <motion.p initial={{ opacity: 0, y:-4 }} animate={{ opacity:1, y:0 }} exit={{ opacity:0 }}
            className="mt-2 flex items-center gap-2 text-xs px-3 py-2 rounded-lg"
            style={{ color: 'var(--accent)', background: 'rgba(139,26,26,0.06)', border: '1px solid rgba(139,26,26,0.15)' }}
          >
            <AlertCircle size={13} /> {error}
          </motion.p>
        )}
      </AnimatePresence>

      <AnimatePresence>
        {files.length > 0 && (
          <motion.div initial={{ opacity:0, y:6 }} animate={{ opacity:1, y:0 }} className="mt-3 space-y-1.5">
            {files.map((f, i) => (
              <div key={i} className="flex items-center justify-between px-4 py-2.5 rounded-xl text-sm"
                style={{ background: 'rgba(74,94,58,0.08)', border: '1px solid rgba(74,94,58,0.2)' }}
              >
                <span className="font-mono text-xs" style={{ color: 'var(--olive)' }}>{f.name}</span>
                <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{formatSize(f.size)}</span>
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
