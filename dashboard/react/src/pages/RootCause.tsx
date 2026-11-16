import { useState, useEffect, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell,
} from 'recharts'
import AlertBadge from '../components/AlertBadge'
import AspectPill, { ASPECT_COLORS } from '../components/AspectPill'

interface AspectEntry { aspect: string; delta: number; direction: string; confidence: string; recon_contrib?: number }
interface ReconBreakdownEntry { aspect: string; contribution_pct: number }
interface Quote { text: string; aspect: string; sentiment: number; stars: number; month: string }
interface DriftEvent { event_month: string; detector: string; top_aspects: AspectEntry[]; recon_breakdown?: ReconBreakdownEntry[]; agreement: string; quotes: Quote[] }
interface ActionItem { aspect: string; severity: number; direction: string; evidence_strength: string }
interface ProductResult {
  parent_asin: string; product_name: string; n_events: number
  events: DriftEvent[]; timeline: Array<{ month: string; aspects: Record<string, number> }>; action_items: ActionItem[]
}
interface AlertEntry { parent_asin: string; event_month: string; state: string; detector: string; lead_time: number | null }

const ACTION_MAP: Record<string, { owner: string; action: string }> = {
  battery: { owner: 'Battery QA', action: 'Review recent hardware revision changes and thermal design' },
  software: { owner: 'Software Engineering', action: 'Check recent OTA/firmware updates and rollback options' },
  connectivity: { owner: 'RF Engineering', action: 'Review baseband firmware and antenna design changes' },
  display: { owner: 'Display QA', action: 'Check panel supplier changes and calibration drift' },
  build: { owner: 'Manufacturing QA', action: 'Review assembly line reports and material sourcing' },
  camera: { owner: 'Camera Team', action: 'Check ISP firmware, lens QC, and sensor calibration' },
  audio: { owner: 'Audio Engineering', action: 'Review speaker/mic component supplier and driver updates' },
  charging: { owner: 'Power Systems', action: 'Check charging IC compatibility and cable certification' },
  performance: { owner: 'Performance Engineering', action: 'Review thermal throttling and memory management' },
}

const ease = [0.16, 1, 0.3, 1] as const
const stagger = {
  container: { animate: { transition: { staggerChildren: 0.06 } } },
  item: { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease } } },
}

export default function RootCause() {
  const [searchParams] = useSearchParams()
  const [explainData, setExplainData] = useState<{ products: ProductResult[] } | null>(null)
  const [alertsData, setAlertsData] = useState<{ alerts: AlertEntry[] } | null>(null)
  const [selectedProduct, setSelectedProduct] = useState(searchParams.get('product') ?? '')
  const [selectedEvent, setSelectedEvent] = useState(0)

  useEffect(() => {
    Promise.all([
      fetch('/data/explainability.json').then(r => r.json()),
      fetch('/data/alerts.json').then(r => r.json()),
    ]).then(([explain, alerts]) => {
      setExplainData(explain)
      setAlertsData(alerts)
      if (!selectedProduct && explain.products.length > 0)
        setSelectedProduct(explain.products[0].parent_asin)
    }).catch(() => {})
  }, [selectedProduct])

  // Find initial event by URL param
  useEffect(() => {
    const monthParam = searchParams.get('month')
    if (monthParam && product) {
      const idx = filteredEvents.findIndex((_ev, i) => {
        const ev = product.events.filter(e => e.detector !== 'PELT')[i]
        return ev?.event_month.startsWith(monthParam)
      })
      if (idx >= 0) setSelectedEvent(idx)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams, explainData])

  const product = useMemo(
    () => explainData?.products.find(p => p.parent_asin === selectedProduct) ?? null,
    [explainData, selectedProduct],
  )

  // Filter out PELT events
  const filteredEvents = useMemo(
    () => product?.events.filter(e => e.detector !== 'PELT') ?? [],
    [product],
  )

  const event = filteredEvents[selectedEvent] ?? null

  // Find alert state for this event
  const alertEntry = useMemo(() => {
    if (!event || !alertsData) return null
    return alertsData.alerts.find(
      a => a.parent_asin === selectedProduct && a.event_month === event.event_month.slice(0, 7)
    ) ?? null
  }, [alertsData, selectedProduct, event])

  if (!explainData || !alertsData) return <div className="text-ink-50">Loading root cause data...</div>

  return (
    <motion.div variants={stagger.container} initial="initial" animate="animate">
      {/* Header */}
      <motion.section variants={stagger.item} className="mb-8">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-3">
          Drift Event Investigation
        </p>
        <h1 className="font-display text-4xl leading-tight mb-2">Root Cause Analysis</h1>
        <p className="text-sm text-ink-100">
          Aspect attribution + LSTM-AE reconstruction error decomposition
        </p>
      </motion.section>

      {/* Product selector */}
      <motion.section variants={stagger.item} className="mb-6">
        <div className="flex flex-wrap gap-2">
          {explainData.products.map(p => (
            <button
              key={p.parent_asin}
              onClick={() => { setSelectedProduct(p.parent_asin); setSelectedEvent(0) }}
              className={`px-4 py-2 rounded-full text-sm transition-all duration-300 ${
                p.parent_asin === selectedProduct
                  ? 'bg-ink-300 text-paper-50 font-medium'
                  : 'bg-paper-50 text-ink-100 border border-paper-300 hover:border-ink-50'
              }`}
            >
              {p.product_name}
            </button>
          ))}
        </div>
      </motion.section>

      {product && filteredEvents.length > 0 && (
        <>
          {/* Event selector */}
          <motion.section variants={stagger.item} className="mb-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-3">Select Event</p>
            <div className="flex gap-2 flex-wrap">
              {filteredEvents.map((ev, i) => {
                const alert = alertsData.alerts.find(
                  a => a.parent_asin === selectedProduct && a.event_month === ev.event_month.slice(0, 7)
                )
                return (
                  <button
                    key={i}
                    onClick={() => setSelectedEvent(i)}
                    className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all duration-200 border ${
                      i === selectedEvent
                        ? 'bg-ink-300/5 border-ink-300/30 text-ink-300 font-medium'
                        : 'bg-paper-50 border-paper-300 text-ink-100 hover:border-ink-50'
                    }`}
                  >
                    {ev.event_month.slice(0, 7)}
                    {alert && <span className="ml-1.5"><AlertBadge state={alert.state} /></span>}
                  </button>
                )
              })}
            </div>
          </motion.section>

          {event && (
            <motion.div
              key={`${selectedProduct}-${selectedEvent}`}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              {/* Event context header */}
              <div className="flex items-center gap-4 mb-6 p-4 rounded-xl bg-paper-50 border border-paper-300">
                {alertEntry && <AlertBadge state={alertEntry.state} size="md" />}
                <div className="text-sm text-ink-200">
                  <span className="font-mono font-medium text-ink-300">{event.event_month.slice(0, 7)}</span>
                  <span className="mx-2 text-paper-400">|</span>
                  <span>{event.detector}</span>
                </div>
                {alertEntry?.lead_time && (
                  <span className="text-xs font-mono text-signal-amber bg-signal-amber-light px-2.5 py-1 rounded-full">
                    LSTM-AE warned {alertEntry.lead_time}m early
                  </span>
                )}
              </div>

              {/* Two-column analysis */}
              <div className="grid md:grid-cols-2 gap-6 mb-8">
                {/* Aspect delta */}
                <div className="bg-paper-50 border border-paper-300 rounded-xl p-6">
                  <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-4">
                    Aspect Sentiment Delta
                  </p>
                  <ResponsiveContainer width="100%" height={240}>
                    <BarChart data={event.top_aspects.map(a => ({ aspect: a.aspect, delta: a.delta }))} layout="vertical" margin={{ left: 8, right: 16 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#E8E5DF" horizontal={false} />
                      <XAxis type="number" tick={{ fontSize: 10, fill: '#6B6B63', fontFamily: 'JetBrains Mono' }} />
                      <YAxis type="category" dataKey="aspect" tick={{ fontSize: 11, fill: '#3D3D37' }} width={85} axisLine={false} tickLine={false} />
                      <Tooltip formatter={(v: number) => v.toFixed(3)} contentStyle={{ backgroundColor: '#FDFCFA', border: '1px solid #E8E5DF', borderRadius: 8, fontSize: 12 }} />
                      <Bar dataKey="delta" radius={[0, 4, 4, 0]} barSize={14} animationDuration={800} animationBegin={200} animationEasing="ease-out">
                        {event.top_aspects.map((a, i) => (
                          <Cell key={i} fill={a.direction === 'deterioration' ? '#D4483B' : '#2D7D6F'} />
                        ))}
                      </Bar>
                      <ReferenceLine x={0} stroke="#9C9C94" strokeDasharray="3 3" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>

                {/* Reconstruction error */}
                <div className="bg-paper-50 border border-paper-300 rounded-xl p-6">
                  <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-4">
                    LSTM-AE Error Decomposition
                  </p>
                  {(() => {
                    // Prefer the full recon_breakdown (all aspects with non-zero LSTM-AE
                    // reconstruction error, sorted desc). Fall back to filtering top_aspects
                    // for older data bundles that predate recon_breakdown.
                    const reconData =
                      event.recon_breakdown && event.recon_breakdown.length > 0
                        ? event.recon_breakdown.map(r => ({ aspect: r.aspect, contribution: r.contribution_pct }))
                        : event.top_aspects
                            .filter(a => a.recon_contrib != null)
                            .map(a => ({ aspect: a.aspect, contribution: a.recon_contrib ?? 0 }))
                    return (
                      <ResponsiveContainer width="100%" height={240}>
                        <BarChart data={reconData} layout="vertical" margin={{ left: 8, right: 16 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#E8E5DF" horizontal={false} />
                          <XAxis type="number" tick={{ fontSize: 10, fill: '#6B6B63', fontFamily: 'JetBrains Mono' }} unit="%" />
                          <YAxis type="category" dataKey="aspect" tick={{ fontSize: 11, fill: '#3D3D37' }} width={85} axisLine={false} tickLine={false} />
                          <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} contentStyle={{ backgroundColor: '#FDFCFA', border: '1px solid #E8E5DF', borderRadius: 8, fontSize: 12 }} />
                          <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={14} animationDuration={800} animationBegin={400} animationEasing="ease-out">
                            {reconData.map((d, i) => (
                              <Cell key={i} fill={ASPECT_COLORS[d.aspect] ?? '#6B8CA3'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    )
                  })()}
                  <div className="mt-2 text-[10px] text-ink-50">
                    Agreement: <span className={
                      event.agreement === 'strong' ? 'text-signal-teal font-medium' :
                      event.agreement === 'partial' ? 'text-signal-amber font-medium' : 'text-ink-50'
                    }>{event.agreement}</span>
                  </div>
                </div>
              </div>

              {/* Review evidence */}
              <div className="mb-8">
                <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-4">Review Evidence</p>
                {event.quotes.length > 0 ? (
                  <div className="grid md:grid-cols-2 gap-4">
                    {event.quotes.map((q, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 8 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.05, duration: 0.3 }}
                        className="p-5 rounded-xl bg-paper-50 border border-paper-300"
                      >
                        <div className="flex items-center gap-2.5 mb-3">
                          <AspectPill aspect={q.aspect} />
                          <span className="text-xs text-ink-50">
                            {'★'.repeat(Math.max(0, Math.min(5, Math.round(q.stars))))}
                            {'☆'.repeat(Math.max(0, 5 - Math.round(q.stars)))}
                          </span>
                          <span className={`text-[10px] font-mono ml-auto ${q.sentiment > 0 ? 'text-signal-teal' : 'text-signal-red'}`}>
                            {q.sentiment > 0 ? '+' : ''}{q.sentiment.toFixed(2)}
                          </span>
                        </div>
                        <p className="text-sm text-ink-200 leading-relaxed">
                          &ldquo;{q.text.slice(0, 220)}{q.text.length > 220 ? '...' : ''}&rdquo;
                        </p>
                        <div className="mt-2 text-[10px] font-mono text-ink-50">{q.month}</div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <p className="text-sm text-ink-50">No review quotes available for this event.</p>
                )}
              </div>

              {/* Recommended actions */}
              {event.top_aspects.length > 0 && (
                <div>
                  <hr className="rule mb-6" />
                  <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-4">
                    Recommended Actions
                  </p>
                  <div className="space-y-2">
                    {event.top_aspects
                      .filter(a => a.direction === 'deterioration')
                      .slice(0, 3)
                      .map((a, i) => {
                        const rec = ACTION_MAP[a.aspect]
                        if (!rec) return null
                        return (
                          <div key={i} className="flex items-start gap-4 p-4 rounded-xl bg-paper-50 border border-paper-300">
                            <div className="w-1.5 h-1.5 rounded-full mt-2 shrink-0 bg-signal-red" />
                            <div className="flex-1">
                              <div className="flex items-center gap-3 mb-1">
                                <AspectPill aspect={a.aspect} size="md" />
                                <span className="font-mono text-xs text-ink-100">
                                  severity {Math.abs(a.delta).toFixed(3)}
                                </span>
                              </div>
                              <p className="text-sm text-ink-300 font-medium mb-0.5">
                                Escalate to {rec.owner}
                              </p>
                              <p className="text-xs text-ink-100">{rec.action}</p>
                            </div>
                          </div>
                        )
                      })}
                  </div>
                  <p className="text-[10px] text-ink-50 mt-3 italic">
                    Recommendations are rule-based heuristics. Human review required before action.
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </>
      )}
    </motion.div>
  )
}
