import { useRef, useEffect } from 'react'
import gsap from 'gsap'

const STATE_COLORS: Record<string, string> = {
  CRITICAL: '#991B1B',
  ALERT: '#2D7D6F',
  WATCH: '#C4880D',
}

interface DriftMarkerProps {
  /** Recharts injects these via customized layer */
  formattedGraphicalItems?: Array<{
    props: { points: Array<{ x: number; y: number }> }
  }>
  /** Alert data keyed by month string */
  alerts: Record<string, { state: string; top_aspect?: string; top_delta?: number; detector?: string }>
  /** Chart data array with month field */
  data: Array<{ month: string }>
  /** X-axis config for coordinate mapping */
  xAxisMap?: Record<string, { x: number; width: number; domain: string[] }>
  /** Height of the chart area */
  height?: number
  /** Offset from top */
  offset?: { top: number; left: number }
}

/**
 * Animated drift event markers rendered as a Recharts customized layer.
 * Renders pulsing dots + vertical lines at each drift event month.
 */
export default function DriftMarker({
  alerts,
  data,
  xAxisMap,
  height = 380,
  offset,
}: DriftMarkerProps) {
  const groupRef = useRef<SVGGElement>(null)

  // Animate markers on mount / data change
  useEffect(() => {
    if (!groupRef.current) return
    const markers = groupRef.current.querySelectorAll('.drift-marker')
    if (markers.length === 0) return

    const ctx = gsap.context(() => {
      gsap.from(markers, {
        autoAlpha: 0,
        scaleY: 0,
        transformOrigin: 'top center',
        stagger: 0.08,
        duration: 0.5,
        delay: 1.5, // wait for chart lines to draw
        ease: 'power3.out',
      })
    }, groupRef.current)

    return () => ctx.revert()
  }, [alerts, data])

  if (!xAxisMap || !data || data.length === 0) return null

  // Get x-axis info to map months to coordinates
  const xAxis = Object.values(xAxisMap)[0]
  if (!xAxis) return null

  const top = offset?.top ?? 8
  const left = offset?.left ?? 0
  const chartWidth = xAxis.width
  const chartX = xAxis.x

  return (
    <g ref={groupRef}>
      {data.map((d, i) => {
        const alert = alerts[d.month]
        if (!alert) return null

        const x = chartX + left + (i / Math.max(data.length - 1, 1)) * chartWidth
        const color = STATE_COLORS[alert.state] ?? '#9C9C94'

        return (
          <g key={d.month} className="drift-marker" style={{ cursor: 'pointer' }}>
            {/* Vertical line */}
            <line
              x1={x} y1={top} x2={x} y2={top + height - 20}
              stroke={color}
              strokeWidth={alert.state === 'CRITICAL' ? 2 : 1.5}
              strokeDasharray={alert.state === 'WATCH' ? '6 3' : undefined}
              opacity={0.7}
            />
            {/* Pulsing dot */}
            <circle cx={x} cy={top + 6} r={4} fill={color} className="drift-pulse" />
            {/* State label */}
            <text
              x={x} y={top - 4}
              textAnchor="middle" fontSize="8" fontWeight="600"
              fontFamily="JetBrains Mono" fill={color}
            >
              {alert.state}
            </text>
          </g>
        )
      })}
    </g>
  )
}
