import { useState, useEffect, useMemo, useCallback } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea, Brush,
} from 'recharts'
import AspectPill, { ASPECT_COLORS } from '../components/AspectPill'
import CusumExplainer from '../components/CusumExplainer'

interface TimelineEntry { month: string; aspects: Record<string, number> }
interface DriftEvent {
  event_month: string; detector: string
  top_aspects: Array<{ aspect: string; delta: number; direction: string }>
}
interface ProductResult {
  parent_asin: string; product_name: string; n_events: number
  events: DriftEvent[]; timeline: TimelineEntry[]
}
interface AlertEntry {
  parent_asin: string; event_month: string; state: string
  detector: string; top_aspect: string; top_delta: number
  direction: string; lead_time: number | null
}
interface AlertsData { alerts: AlertEntry[] }
interface ExplainData { products: ProductResult[] }

const DETECTOR_COLORS: Record<string, string> = {
  'CUSUM+LSTM-AE': '#991B1B',
  'CUSUM': '#2D7D6F',
  'LSTM-AE': '#C4880D',
}

const DETECTOR_LABEL: Record<string, string> = {
  'CUSUM+LSTM-AE': 'Both',
  'CUSUM': 'CUSUM',
  'LSTM-AE': 'LSTM',
}

const ALL_ASPECTS = Object.keys(ASPECT_COLORS)

/* Exponential Moving Average — smooths noisy monthly sentiment so drift events stand out */
function emaSmooth(
  data: Record<string, unknown>[],
  aspects: string[],
  alpha: number,
): Record<string, unknown>[] {
  const state: Record<string, number> = {}
  return data.map(row => {
    const smoothed: Record<string, unknown> = { month: row.month }
    for (const asp of aspects) {
      const raw = row[asp]
      if (raw == null || typeof raw !== 'number') continue
      if (!(asp in state)) {
        state[asp] = raw
      } else {
        state[asp] = alpha * raw + (1 - alpha) * state[asp]
      }
      smoothed[asp] = Math.round(state[asp] * 1000) / 1000
    }
    return smoothed
  })
}

const ease = [0.16, 1, 0.3, 1] as const
const stagger = {
  container: { animate: { transition: { staggerChildren: 0.06 } } },
  item: { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease } } },
}

export default function ProductPulse() {
  const [searchParams] = useSearchParams()
  const navigate = useNavigate()
  const [explainData, setExplainData] = useState<ExplainData | null>(null)
  const [alertsData, setAlertsData] = useState<AlertsData | null>(null)
  const [selectedAsin, setSelectedAsin] = useState(searchParams.get('product') ?? '')
  const [visibleAspects, setVisibleAspects] = useState<Set<string>>(new Set())
  const [hoveredAspect, setHoveredAspect] = useState<string | null>(null)
  const [showTimeline, setShowTimeline] = useState(false)
  const [smoothing, setSmoothing] = useState(true)
  const [brushRange, setBrushRange] = useState<{ startIndex: number; endIndex: number } | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/data/explainability.json').then(r => r.json()),
      fetch('/data/alerts.json').then(r => r.json()),
    ]).then(([explain, alerts]) => {
      setExplainData(explain)
      setAlertsData(alerts)
      if (!selectedAsin && explain.products.length > 0) {
        setSelectedAsin(explain.products[0].parent_asin)
      }
    }).catch(() => {})
  }, [selectedAsin])

  const product = useMemo(
    () => explainData?.products.find(p => p.parent_asin === selectedAsin) ?? null,
    [explainData, selectedAsin],
  )

  const productAlerts = useMemo(
    () => alertsData?.alerts.filter(a => a.parent_asin === selectedAsin) ?? [],
    [alertsData, selectedAsin],
  )

  /* Top drift aspects ranked by cumulative absolute delta — default shows top 2 */
  const topDriftAspects = useMemo(() => {
    const deltas: Record<string, number> = {}
    for (const a of productAlerts) {
      if (a.top_aspect) {
        deltas[a.top_aspect] = (deltas[a.top_aspect] || 0) + Math.abs(a.top_delta)
      }
    }
    return Object.entries(deltas)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2)
      .map(([asp]) => asp)
  }, [productAlerts])

  /* All drift-involved aspects (for the "Drift" quick-select) */
  const driftAspects = useMemo(() => {
    const aspects = new Set<string>()
    for (const a of productAlerts) {
      if (a.top_aspect) aspects.add(a.top_aspect)
    }
    return aspects
  }, [productAlerts])

  /* Reset to top 2 drift aspects when product changes */
  useEffect(() => {
    if (topDriftAspects.length > 0) {
      setVisibleAspects(new Set(topDriftAspects))
    } else {
      setVisibleAspects(new Set(ALL_ASPECTS.slice(0, 2)))
    }
    setShowTimeline(false)
  }, [selectedAsin, topDriftAspects])

  const alertsByMonth = useMemo(() => {
    const map: Record<string, AlertEntry> = {}
    for (const a of productAlerts) {
      if (!map[a.event_month] || detectorRank(a.detector) < detectorRank(map[a.event_month].detector)) {
        map[a.event_month] = a
      }
    }
    return map
  }, [productAlerts])

  const chartData = useMemo(() => {
    if (!product) return []
    return product.timeline.map(t => ({ month: t.month.slice(0, 7), ...t.aspects }))
  }, [product])

  /* Reset brush to the last 24 months on product switch or timeline-length change.
     Depend on chartData.length (not chartData) so smoothing toggles preserve the range. */
  useEffect(() => {
    if (chartData.length === 0) { setBrushRange(null); return }
    const endIndex = chartData.length - 1
    const startIndex = Math.max(0, chartData.length - 24)
    setBrushRange({ startIndex, endIndex })
  }, [selectedAsin, chartData.length])

  const displayData = useMemo(() => {
    if (!smoothing) return chartData
    return emaSmooth(chartData, ALL_ASPECTS, 0.4)
  }, [chartData, smoothing])

  const toggleAspect = (asp: string) => {
    setVisibleAspects(prev => {
      const next = new Set(prev)
      if (next.has(asp)) next.delete(asp); else next.add(asp)
      return next
    })
  }

  const showAllAspects = useCallback(() => setVisibleAspects(new Set(ALL_ASPECTS)), [])
  const clearAspects = useCallback(() => setVisibleAspects(new Set()), [])
  const showDriftOnly = useCallback(() => {
    setVisibleAspects(driftAspects.size > 0 ? new Set(driftAspects) : new Set(ALL_ASPECTS.slice(0, 3)))
  }, [driftAspects])

  if (!explainData || !alertsData) return <div className="text-ink-50">Loading product data...</div>

  const aspectCount = visibleAspects.size
  const isDriftDefault = driftAspects.size > 0 && visibleAspects.size === driftAspects.size
    && [...driftAspects].every(a => visibleAspects.has(a))
  const isAllSelected = aspectCount === ALL_ASPECTS.length

  const detectorCounts = (() => {
    const counts: Record<string, number> = {}
    for (const a of productAlerts) {
      const label = DETECTOR_LABEL[a.detector] ?? a.detector
      counts[label] = (counts[label] || 0) + 1
    }
    return counts
  })()

  /* Default brush range: last 24 months */
  const brushEnd = chartData.length - 1
  const brushStart = Math.max(0, chartData.length - 24)

  return (
    <motion.div variants={stagger.container} initial="initial" animate="animate">
      {/* Header */}
      <motion.section variants={stagger.item} className="mb-8">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-3">
          Product Quality Monitor
        </p>
        <h1 className="font-display text-4xl leading-tight mb-2">Product Pulse</h1>
        <p className="text-sm text-ink-100">
          Aspect sentiment over time &middot; drift events highlighted
          {smoothing && <span className="text-ink-50"> &middot; EMA smoothed</span>}
        </p>
      </motion.section>

      {/* Product selector */}
      <motion.section variants={stagger.item} className="mb-6">
        <div className="flex flex-wrap gap-2">
          {explainData.products.map(p => (
            <button
              key={p.parent_asin}
              onClick={() => setSelectedAsin(p.parent_asin)}
              className={`px-4 py-2 rounded-full text-sm transition-all duration-300 ${
                p.parent_asin === selectedAsin
                  ? 'bg-ink-300 text-paper-50 font-medium'
                  : 'bg-paper-50 text-ink-100 border border-paper-300 hover:border-ink-50'
              }`}
            >
              {p.product_name}
            </button>
          ))}
        </div>
      </motion.section>

      {product && (
        <>
          {/* Aspect toggles */}
          <motion.section variants={stagger.item} className="mb-5">
            <div className="flex items-center gap-3 mb-2">
              <div className="flex gap-1">
                <button
                  onClick={showDriftOnly}
                  className={`px-2.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider transition-colors ${
                    isDriftDefault && !isAllSelected
                      ? 'bg-ink-200 text-paper-50'
                      : 'text-ink-50 hover:text-ink-200'
                  }`}
                >
                  Drift
                </button>
                <button
                  onClick={showAllAspects}
                  className={`px-2.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider transition-colors ${
                    isAllSelected
                      ? 'bg-ink-200 text-paper-50'
                      : 'text-ink-50 hover:text-ink-200'
                  }`}
                >
                  All
                </button>
                <button
                  onClick={clearAspects}
                  className={`px-2.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider transition-colors ${
                    aspectCount === 0
                      ? 'bg-ink-200 text-paper-50'
                      : 'text-ink-50 hover:text-ink-200'
                  }`}
                >
                  Clear
                </button>
              </div>
              <span className="w-px h-4 bg-paper-300" />
              <button
                onClick={() => setSmoothing(s => !s)}
                className={`px-2.5 py-0.5 rounded text-[10px] font-mono uppercase tracking-wider transition-colors ${
                  smoothing
                    ? 'bg-ink-200 text-paper-50'
                    : 'text-ink-50 hover:text-ink-200'
                }`}
              >
                {smoothing ? 'Smoothed' : 'Raw'}
              </button>
              <span className="text-[10px] text-ink-50 font-mono">
                {aspectCount} of {ALL_ASPECTS.length} aspects
              </span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {Object.entries(ASPECT_COLORS).map(([asp, color]) => (
                <button
                  key={asp}
                  onClick={() => toggleAspect(asp)}
                  onMouseEnter={() => setHoveredAspect(asp)}
                  onMouseLeave={() => setHoveredAspect(null)}
                  className={`px-3 py-1 rounded-full text-xs font-medium transition-all duration-200 ${
                    visibleAspects.has(asp) ? 'text-white shadow-sm' : 'text-ink-50 bg-paper-200'
                  }`}
                  style={visibleAspects.has(asp) ? { backgroundColor: color } : {}}
                >
                  {asp}
                </button>
              ))}
            </div>
            <p className="text-[10px] text-ink-50/60 mt-2 font-mono">
              Hover aspects to highlight &middot; Drag range below chart to zoom
            </p>
          </motion.section>

          {/* Main chart */}
          <motion.section variants={stagger.item} className="mb-10">
            <div className="bg-paper-50 border border-paper-300 rounded-xl p-6">
              {/* Detector & drift legend */}
              <div className="flex items-center gap-4 mb-4 text-[10px] text-ink-50">
                <span className="font-mono uppercase tracking-wider">Detected by</span>
                <span className="flex items-center gap-1">
                  <span className="w-4 h-0" style={{ borderTop: '2px solid #2D7D6F' }} />CUSUM
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-4 h-0" style={{ borderTop: '2px dashed #C4880D' }} />LSTM
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-4 h-0" style={{ borderTop: '2.5px solid #991B1B' }} />Both
                </span>
                <span className="w-px h-3 bg-paper-300 mx-1" />
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#D4483B', opacity: 0.25 }} />Deterioration
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: '#2D7D6F', opacity: 0.25 }} />Improvement
                </span>
              </div>

              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={displayData} margin={{ top: 36, right: 16, left: 0, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E8E5DF" />
                  <XAxis dataKey="month" tick={{ fontSize: 9, fill: '#6B6B63', fontFamily: 'JetBrains Mono' }} interval="preserveStartEnd" />
                  <YAxis domain={[-1.2, 1.2]} tick={{ fontSize: 10, fill: '#6B6B63', fontFamily: 'JetBrains Mono' }} axisLine={false} />
                  <Tooltip
                    content={({ active, payload, label }) => {
                      if (!active || !payload) return null
                      const visible = payload.filter(p => visibleAspects.has(p.dataKey as string))
                      if (visible.length === 0) return null
                      const rawRow = smoothing ? chartData.find(d => d.month === label) as Record<string, unknown> | undefined : null
                      return (
                        <div style={{ backgroundColor: '#FDFCFA', border: '1px solid #E8E5DF', borderRadius: 8, padding: '8px 12px', fontSize: 11, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}>
                          <p style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: '#6B6B63', marginBottom: 4 }}>
                            {label}
                            {smoothing && <span style={{ color: '#9C9C94', marginLeft: 6 }}>ema</span>}
                          </p>
                          {visible
                            .sort((a, b) => Math.abs(b.value as number) - Math.abs(a.value as number))
                            .map(entry => {
                              const rawVal = rawRow?.[entry.dataKey as string]
                              return (
                                <div key={entry.dataKey as string} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '1px 0' }}>
                                  <span style={{ width: 8, height: 8, borderRadius: '50%', backgroundColor: entry.color, flexShrink: 0 }} />
                                  <span style={{ color: '#3D3D37' }}>{entry.dataKey}</span>
                                  <span style={{ fontFamily: 'JetBrains Mono', color: '#1B1B18', marginLeft: 'auto', fontWeight: 500 }}>
                                    {(entry.value as number).toFixed(3)}
                                  </span>
                                  {rawVal != null && typeof rawVal === 'number' && (
                                    <span style={{ fontFamily: 'JetBrains Mono', color: '#9C9C94', fontSize: 10 }}>
                                      ({(rawVal as number).toFixed(2)})
                                    </span>
                                  )}
                                </div>
                              )
                            })}
                        </div>
                      )
                    }}
                  />
                  <ReferenceLine y={0} stroke="#D4D0C8" />

                  {Object.entries(ASPECT_COLORS).map(([asp, color]) => (
                    <Line
                      key={asp} type="monotone" dataKey={asp}
                      stroke={visibleAspects.has(asp) ? color : 'transparent'}
                      strokeWidth={
                        !visibleAspects.has(asp) ? 0
                          : hoveredAspect === asp ? 3
                          : hoveredAspect !== null && visibleAspects.has(hoveredAspect) ? 1
                          : 2
                      }
                      strokeOpacity={
                        !visibleAspects.has(asp) ? 0
                          : hoveredAspect === asp ? 1
                          : hoveredAspect !== null && visibleAspects.has(hoveredAspect) ? 0.2
                          : 1
                      }
                      dot={false}
                      activeDot={visibleAspects.has(asp)
                        ? { r: 4, stroke: color, strokeWidth: 2, fill: '#FDFCFA' }
                        : false
                      }
                      connectNulls
                      animationDuration={800}
                      animationEasing="ease-out"
                    />
                  ))}

                  {/* Drift zones — shaded by direction & magnitude */}
                  {Object.entries(alertsByMonth).map(([month, alert]) => {
                    const idx = displayData.findIndex(d => d.month === month)
                    const x1 = (displayData[Math.max(0, idx - 3)]?.month as string) ?? month
                    const x2 = month
                    const isDet = alert.direction === 'deterioration'
                    const opacity = Math.min(0.2, Math.max(0.05, Math.abs(alert.top_delta) * 0.12))
                    return (
                      <ReferenceArea
                        key={`zone-${month}`}
                        x1={x1}
                        x2={x2}
                        fill={isDet ? '#D4483B' : '#2D7D6F'}
                        fillOpacity={opacity}
                        stroke="none"
                        ifOverflow="extendDomain"
                      />
                    )
                  })}

                  {/* Drift markers — labelled by detector (staggered to avoid overlap) */}
                  {(() => {
                    const sorted = Object.entries(alertsByMonth)
                      .map(([month, alert]) => ({ month, alert, idx: displayData.findIndex(d => d.month === month) }))
                      .sort((a, b) => a.idx - b.idx)
                    const dyMap = new Map<string, number>()
                    for (let i = 0; i < sorted.length; i++) {
                      if (i === 0) { dyMap.set(sorted[i].month, 0); continue }
                      const gap = sorted[i].idx - sorted[i - 1].idx
                      const prevDy = dyMap.get(sorted[i - 1].month) ?? 0
                      dyMap.set(sorted[i].month, gap <= 3 ? (prevDy === 0 ? -14 : 0) : 0)
                    }
                    return sorted.map(({ month, alert }) => {
                      const isDet = alert.direction === 'deterioration'
                      const dy = dyMap.get(month) ?? 0
                      const labelText = `${alert.top_aspect} Δ${alert.top_delta > 0 ? '+' : ''}${alert.top_delta.toFixed(2)}`
                      const labelFill = isDet ? '#991B1B' : '#1B6B5F'
                      return (
                        <ReferenceLine
                          key={month}
                          x={month}
                          stroke={DETECTOR_COLORS[alert.detector] ?? '#9C9C94'}
                          strokeDasharray={alert.detector === 'LSTM-AE' ? '6 3' : undefined}
                          strokeWidth={alert.detector === 'CUSUM+LSTM-AE' ? 2.5 : 1.5}
                          label={({ viewBox }: { viewBox: { x: number; y: number } }) => (
                            <text
                              x={viewBox.x}
                              y={viewBox.y + dy}
                              dy={-6}
                              textAnchor="middle"
                              fontSize={9}
                              fontWeight={700}
                              fontFamily="JetBrains Mono"
                              fill={labelFill}
                            >
                              {labelText}
                            </text>
                          )}
                        />
                      )
                    })
                  })()}

                  <Brush
                    dataKey="month"
                    height={24}
                    startIndex={brushRange?.startIndex ?? brushStart}
                    endIndex={brushRange?.endIndex ?? brushEnd}
                    onChange={(range) => {
                      if (!range) return
                      const { startIndex, endIndex } = range
                      if (typeof startIndex !== 'number' || typeof endIndex !== 'number') return
                      setBrushRange(prev =>
                        prev && prev.startIndex === startIndex && prev.endIndex === endIndex
                          ? prev
                          : { startIndex, endIndex }
                      )
                    }}
                    stroke="#D4D0C8"
                    fill="#FAFAF7"
                    travellerWidth={8}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </motion.section>

          {/* CUSUM Methodology */}
          <motion.section variants={stagger.item} className="mb-2">
            <hr className="rule mb-8" />
            <CusumExplainer />
          </motion.section>

          {/* Collapsible event timeline */}
          <motion.section variants={stagger.item}>
            <hr className="rule mb-6" />
            <button
              onClick={() => setShowTimeline(prev => !prev)}
              className="flex items-center gap-3 mb-5 w-full text-left"
            >
              <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50">
                Alert Timeline &middot; {productAlerts.length} events
              </p>
              <div className="flex items-center gap-2">
                {Object.entries(detectorCounts).map(([label, count]) => (
                  <span key={label} className="text-[10px] font-mono text-ink-100">
                    {count} {label}
                  </span>
                ))}
              </div>
              <span className={`text-[10px] text-ink-50 font-mono ml-auto transition-transform duration-300 ${showTimeline ? 'rotate-180' : ''}`}>
                &#x25BC;
              </span>
            </button>

            <AnimatePresence>
              {showTimeline && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.35, ease: [0.16, 1, 0.3, 1] }}
                  className="overflow-hidden"
                >
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {productAlerts.map((alert, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 12 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.04, duration: 0.3 }}
                        onClick={() => navigate(`/root-cause?product=${alert.parent_asin}&month=${alert.event_month}`)}
                        className="p-4 rounded-xl bg-paper-50 border border-paper-300 hover:shadow-sm hover:border-paper-400 transition-all cursor-pointer"
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-mono text-sm font-medium text-ink-300">{alert.event_month}</span>
                          <span
                            className="inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[10px] font-medium font-mono uppercase tracking-wide"
                            style={{
                              color: DETECTOR_COLORS[alert.detector] ?? '#6B6B63',
                              borderColor: `${DETECTOR_COLORS[alert.detector] ?? '#6B6B63'}40`,
                              backgroundColor: `${DETECTOR_COLORS[alert.detector] ?? '#6B6B63'}15`,
                            }}
                          >
                            {DETECTOR_LABEL[alert.detector] ?? alert.detector}
                          </span>
                        </div>
                        <div className="flex items-center gap-2 mb-1.5">
                          {alert.top_aspect && <AspectPill aspect={alert.top_aspect} />}
                          <span className={`font-mono text-xs ${alert.direction === 'deterioration' ? 'text-signal-red' : 'text-signal-teal'}`}>
                            {alert.top_delta > 0 ? '+' : ''}{alert.top_delta.toFixed(2)}
                          </span>
                        </div>
                        {alert.lead_time && (
                          <p className="text-[10px] text-signal-amber font-mono">
                            LSTM-AE warned {alert.lead_time}m before CUSUM confirmed
                          </p>
                        )}
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.section>
        </>
      )}
    </motion.div>
  )
}

function detectorRank(detector: string): number {
  if (detector === 'CUSUM+LSTM-AE') return 0
  if (detector === 'CUSUM' || detector === 'PELT') return 1
  if (detector === 'LSTM-AE') return 2
  return 3
}
