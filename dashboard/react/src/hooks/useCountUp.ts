import { useState, useEffect, useRef } from 'react'

export function useCountUp(
  target: number,
  duration = 1200,
  startOnMount = true,
): number {
  const [value, setValue] = useState(0)
  const ref = useRef<number>(0)

  useEffect(() => {
    if (!startOnMount) return
    const start = performance.now()
    const step = (now: number) => {
      const progress = Math.min((now - start) / duration, 1)
      const eased = 1 - Math.pow(1 - progress, 3)
      const current = Math.round(eased * target)
      setValue(current)
      if (progress < 1) {
        ref.current = requestAnimationFrame(step)
      }
    }
    ref.current = requestAnimationFrame(step)
    return () => cancelAnimationFrame(ref.current)
  }, [target, duration, startOnMount])

  return value
}
