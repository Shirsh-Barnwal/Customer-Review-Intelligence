import { useRef } from 'react'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'

const REDUCED_MOTION = typeof window !== 'undefined'
  && window.matchMedia('(prefers-reduced-motion: reduce)').matches

/** Editorial easing — smooth deceleration, no bounce */
export const EASE = 'power3.out'
export const DURATION = REDUCED_MOTION ? 0 : 0.6

/**
 * Creates a GSAP timeline scoped to a container ref.
 * Timeline auto-plays on mount and is reverted on unmount.
 * All string selectors inside `setup` are scoped to the container.
 */
export function useSlideTimeline(
  setup: (tl: gsap.core.Timeline, el: HTMLDivElement) => void,
) {
  const containerRef = useRef<HTMLDivElement>(null)

  useGSAP(() => {
    if (!containerRef.current) return
    const tl = gsap.timeline({
      defaults: { ease: EASE, duration: DURATION },
    })
    setup(tl, containerRef.current)
  }, { scope: containerRef })

  return containerRef
}
