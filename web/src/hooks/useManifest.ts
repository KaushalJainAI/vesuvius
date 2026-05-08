import { useEffect, useState } from 'react'
import type { Manifest } from '@/types/segment'

export function useManifest() {
  const [manifest, setManifest] = useState<Manifest | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch('/manifest.json')
      .then(r => r.json())
      .then(data => { setManifest(data); setLoading(false) })
      .catch(() => setLoading(false))
  }, [])

  return { manifest, loading }
}
