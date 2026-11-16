import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { useNavigate } from 'react-router-dom'
import AlertBadge from '../components/AlertBadge'
import AspectPill from '../components/AspectPill'
import { useCountUp } from '../hooks/useCountUp'

interface Alert {
  parent_asin: string
  product_name: string
  event_month: string
  state: string
  type: string
  detector: string
  top_aspect: string
  top_delta: number
  direction: string
  review_count: number
  lead_time: number | null
}

interface PortfolioEntry {
  parent_asin: string
  product_name: string
  health: string
  active_alerts: number
  critical: number
  alerts: number
  watches: number
  top_issue: string | null
}

interface AlertsData {
  alerts: Alert[]
  portfolio: PortfolioEntry[]
}

const ease = [0.16, 1, 0.3, 1] as const
const stagger = {
  container: { animate: { transition: { staggerChildren: 0.06 } } },
  item: { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease } } },
}

const HEALTH_STYLES: Record<string, string> = {
  red: 'bg-signal-red',
  amber: 'bg-signal-amber',
  green: 'bg-signal-teal',
}

export default function CommandCenter() {
  const [data, setData] = useState<AlertsData | null>(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetch('/data/alerts.json')
      .then(r => r.json())
      .then(setData)
      .catch(() => {})
  }, [])

  /* Count unique products by worst severity */
  const worstByProduct = new Map<string, string>()
  const SEVERITY: Record<string, number> = { WATCH: 1, ALERT: 2, CRITICAL: 3 }
  for (const a of data?.alerts ?? []) {
    const prev = worstByProduct.get(a.parent_asin)
    if (!prev || (SEVERITY[a.state] ?? 0) > (SEVERITY[prev] ?? 0)) {
      worstByProduct.set(a.parent_asin, a.state)
    }
  }
  let criticalCount = 0, alertCount = 0, watchCount = 0
  for (const state of worstByProduct.values()) {
    if (state === 'CRITICAL') criticalCount++
    else if (state === 'ALERT') alertCount++
    else if (state === 'WATCH') watchCount++
  }

  /* Per-product detector event counts for portfolio cards */
  const detectorsByProduct = new Map<string, { cusum: number; lstm: number; both: number }>()
  for (const a of data?.alerts ?? []) {
    let entry = detectorsByProduct.get(a.parent_asin)
    if (!entry) { entry = { cusum: 0, lstm: 0, both: 0 }; detectorsByProduct.set(a.parent_asin, entry) }
    if (a.detector === 'CUSUM+LSTM-AE') entry.both++
    else if (a.detector === 'CUSUM') entry.cusum++
    else if (a.detector === 'LSTM-AE') entry.lstm++
  }

  const criticalUp = useCountUp(criticalCount, 800)
  const alertUp = useCountUp(alertCount, 900)
  const watchUp = useCountUp(watchCount, 700)

  if (!data) return <div className="text-ink-50">Loading command center...</div>

  return (
    <motion.div variants={stagger.container} initial="initial" animate="animate">
      {/* Header */}
      <motion.section variants={stagger.item} className="mb-10">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-3">
          IntelliSys Product Quality Workspace
        </p>
        <h1 className="font-display text-4xl leading-tight mb-2">
          Command Center
        </h1>
        <p className="text-sm text-ink-100">
          Monitoring 328 products across 312,100 reviews &middot; Amazon 2015&ndash;2020
        </p>
      </motion.section>

      {/* Alert summary strip */}
      <motion.section variants={stagger.item} className="mb-10">
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: 'Critical', value: criticalUp, color: 'text-red-900', bg: 'bg-red-900/5 border-red-900/15' },
            { label: 'Alert', value: alertUp, color: 'text-signal-red', bg: 'bg-signal-red/5 border-signal-red/15' },
            { label: 'Watch', value: watchUp, color: 'text-signal-amber', bg: 'bg-signal-amber/5 border-signal-amber/15' },
          ].map(s => (
            <div key={s.label} className={`p-5 rounded-xl border ${s.bg}`}>
              <div className={`stat-value text-3xl ${s.color} mb-1`}>{s.value}</div>
              <div className="text-xs text-ink-100 font-medium">{s.label}</div>
            </div>
          ))}
        </div>
      </motion.section>

      {/* Portfolio health */}
      <motion.section variants={stagger.item} className="mb-12">
        <hr className="rule mb-6" />
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-5">
          Portfolio Health
        </p>
        <div className="grid md:grid-cols-5 gap-3">
          {data.portfolio.map(p => {
            const d = detectorsByProduct.get(p.parent_asin)
            return (
              <motion.div
                key={p.parent_asin}
                variants={stagger.item}
                onClick={() => navigate(`/pulse?product=${p.parent_asin}`)}
                className="p-4 rounded-xl bg-paper-50 border border-paper-300 hover:border-paper-400 hover:shadow-sm transition-all cursor-pointer"
              >
                <div className="flex items-start justify-between mb-2">
                  <h4 className="text-sm font-medium text-ink-300 leading-snug pr-2">{p.product_name}</h4>
                  <div className={`w-2.5 h-2.5 rounded-full shrink-0 mt-0.5 ${HEALTH_STYLES[p.health]}`} />
                </div>
                <div className="flex items-baseline gap-1.5 mb-2">
                  <span className="stat-value text-2xl text-ink-300">{p.active_alerts}</span>
                  <span className="text-[10px] text-ink-50">events</span>
                </div>
                {d && (
                  <div className="flex gap-1.5 text-[9px] font-mono text-ink-50">
                    {d.both > 0 && <span className="text-red-900">{d.both} Both</span>}
                    {d.cusum > 0 && <span className="text-signal-teal">{d.cusum} CUSUM</span>}
                    {d.lstm > 0 && <span className="text-signal-amber">{d.lstm} LSTM</span>}
                  </div>
                )}
                {p.top_issue && (
                  <div className="mt-2">
                    <AspectPill aspect={p.top_issue} />
                  </div>
                )}
              </motion.div>
            )
          })}
        </div>
      </motion.section>

      {/* Triage queue */}
      <motion.section variants={stagger.item}>
        <hr className="rule mb-6" />
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-5">
          Triage Queue &middot; {data.alerts.length} events
        </p>

        <div className="bg-paper-50 border border-paper-300 rounded-xl overflow-hidden">
          {/* Header row */}
          <div className="grid grid-cols-[1fr_100px_100px_120px_80px_80px_80px] gap-3 px-5 py-3 border-b border-paper-300 text-[10px] font-mono uppercase tracking-wide text-ink-50">
            <span>Product</span>
            <span>Status</span>
            <span>Type</span>
            <span>Top Aspect</span>
            <span>Delta</span>
            <span>Month</span>
            <span>Lead Time</span>
          </div>

          {/* Alert rows */}
          {data.alerts.map((alert, i) => (
            <motion.div
              key={`${alert.parent_asin}-${alert.event_month}-${i}`}
              variants={stagger.item}
              onClick={() => navigate(`/pulse?product=${alert.parent_asin}`)}
              className="grid grid-cols-[1fr_100px_100px_120px_80px_80px_80px] gap-3 px-5 py-3.5 border-b border-paper-200 last:border-0 hover:bg-paper-100 cursor-pointer transition-colors items-center"
            >
              <div>
                <span className="text-sm font-medium text-ink-300">{alert.product_name}</span>
              </div>
              <div><AlertBadge state={alert.state} /></div>
              <div className="text-[10px] text-ink-100 font-mono">
                {alert.detector === 'CUSUM' ? 'Persistent' : 'Transient'}
              </div>
              <div>
                {alert.top_aspect && <AspectPill aspect={alert.top_aspect} />}
              </div>
              <div className={`font-mono text-xs ${alert.direction === 'deterioration' ? 'text-signal-red' : 'text-signal-teal'}`}>
                {alert.top_delta > 0 ? '+' : ''}{alert.top_delta.toFixed(2)}
              </div>
              <div className="font-mono text-xs text-ink-100">{alert.event_month}</div>
              <div className="font-mono text-xs text-ink-50">
                {alert.lead_time ? `${alert.lead_time}m early` : '\u2014'}
              </div>
            </motion.div>
          ))}
        </div>
      </motion.section>
    </motion.div>
  )
}
