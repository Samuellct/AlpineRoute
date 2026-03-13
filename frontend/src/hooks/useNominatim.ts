// geocoding Nominatim avec debounce
import { useState, useEffect, useRef } from 'react'
import type { NominatimResult } from '../types'

export function useNominatim(query: string, enabled: boolean) {
  const [results, setResults] = useState<NominatimResult[]>([])
  const [loading, setLoading] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!enabled || query.length < 3) {
      setResults([])
      return
    }

    if (timerRef.current) clearTimeout(timerRef.current)

    timerRef.current = setTimeout(async () => {
      setLoading(true)
      try {
        const url = `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=5&countrycodes=fr,it,ch,at,de,si`
        const resp = await fetch(url, {
          headers: { 'User-Agent': 'AlpineRoute/2.0.0' },
        })
        if (resp.ok) {
          const data = await resp.json()
          setResults(data)
        }
      } catch {
        // silencieux
      } finally {
        setLoading(false)
      }
    }, 500)

    return () => { if (timerRef.current) clearTimeout(timerRef.current) }
  }, [query, enabled])

  return { results, loading }
}
