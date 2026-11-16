import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend, Cell,
} from 'recharts'
import { ASPECT_COLORS } from '../components/AspectPill'

interface ComparisonRow { Method: string; Recall: number; 'Mean Delay': number; 'Mean Events/Product': number }
interface LstmParams { hidden_dim: number; latent_dim: number; seq_len: number; n_seeds: number; threshold: number; anomaly_rate_drift: number; anomaly_rate_normal: number }
interface DriftMetrics { comparison: ComparisonRow[]; by_drift_type: Array<Record<string, unknown>>; lstm_ae: LstmParams; n_synthetic_cases: number; n_normal_train: number; n_drift_products: number }
interface AbsaMetrics {
  cv_dl_macro_f1: number; cv_lr_macro_f1: number
  cv_dl_macro_f1_ci_95: [number, number]; cv_lr_macro_f1_ci_95: [number, number]
  cv_dl_per_aspect_f1: Record<string, number>; cv_lr_per_aspect_f1: Record<string, number>
  cv_n_triplets: number; cv_n_folds: number
}

const METHOD_COLORS: Record<string, string> = { 'CUSUM': '#2D7D6F', 'LSTM-AE': '#D4483B' }

const ease = [0.16, 1, 0.3, 1] as const
const stagger = {
  container: { animate: { transition: { staggerChildren: 0.06 } } },
  item: { initial: { opacity: 0, y: 20 }, animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease } } },
}

export default function SystemHealth() {
  const [drift, setDrift] = useState<DriftMetrics | null>(null)
  const [absa, setAbsa] = useState<AbsaMetrics | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/data/drift_metrics.json').then(r => r.json()),
      fetch('/data/absa_metrics.json').then(r => r.json()),
    ]).then(([d, a]) => { setDrift(d); setAbsa(a) }).catch(() => {})
  }, [])

  if (!drift || !absa) return <div className="text-ink-50">Loading system health...</div>

  // Only CUSUM and LSTM-AE
  const activeDetectors = drift.comparison.filter(c => c.Method === 'CUSUM' || c.Method === 'LSTM-AE')
  const retiredDetectors = drift.comparison.filter(c => c.Method !== 'CUSUM' && c.Method !== 'LSTM-AE')

  const byTypeData = drift.by_drift_type.map(row => ({
    type: row['Drift Type'] as string,
    'CUSUM': row['CUSUM'] as number,
    'LSTM-AE': row['LSTM-AE'] as number,
  }))

  const perAspectData = Object.keys(ASPECT_COLORS)
    .map(asp => ({ aspect: asp, dl: absa.cv_dl_per_aspect_f1[asp] ?? 0, lr: absa.cv_lr_per_aspect_f1[asp] ?? 0 }))
    .sort((a, b) => b.dl - a.dl)

  return (
    <motion.div variants={stagger.container} initial="initial" animate="animate">
      {/* Header */}
      <motion.section variants={stagger.item} className="mb-10">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-3">
          Model Evaluation & Methodology
        </p>
        <h1 className="font-display text-4xl leading-tight mb-2">System Health</h1>
        <p className="text-sm text-ink-100">
          Detector performance, ABSA validation, architecture details, and known limitations
        </p>
      </motion.section>

      {/* Detector performance */}
      <motion.section variants={stagger.item} className="mb-12">
        <hr className="rule mb-6" />
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-5">
          Active Detectors
        </p>

        <div className="grid md:grid-cols-2 gap-4 mb-6">
          {activeDetectors.map(m => (
            <div key={m.Method} className="p-6 rounded-xl bg-paper-50 border border-paper-300">
              <div className="flex items-center gap-3 mb-4">
                <span className="w-3 h-3 rounded-full" style={{ backgroundColor: METHOD_COLORS[m.Method] }} />
                <span className="font-mono text-sm font-medium" style={{ color: METHOD_COLORS[m.Method] }}>{m.Method}</span>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <div className="stat-value text-3xl text-ink-300">{(m.Recall * 100).toFixed(1)}%</div>
                  <div className="text-[10px] text-ink-50 mt-0.5">Recall</div>
                </div>
                <div>
                  <div className="stat-value text-3xl text-ink-300">{m['Mean Delay'].toFixed(2)}</div>
                  <div className="text-[10px] text-ink-50 mt-0.5">Mean Delay (m)</div>
                </div>
                <div>
                  <div className="stat-value text-3xl text-ink-300">{m['Mean Events/Product'].toFixed(1)}</div>
                  <div className="text-[10px] text-ink-50 mt-0.5">Events/Product</div>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Retired methods note */}
        <div className="p-4 rounded-xl bg-paper-200/40 border border-paper-300 text-xs text-ink-100">
          <span className="font-medium text-ink-200">Evaluated and excluded:</span>{' '}
          {retiredDetectors.map(m => `${m.Method} (${(m.Recall * 100).toFixed(1)}% recall)`).join(', ')}.
          These methods were benchmarked on {drift.n_synthetic_cases} semi-synthetic cases and did not meet
          production performance thresholds.
        </div>
      </motion.section>

      {/* By drift type */}
      <motion.section variants={stagger.item} className="mb-12">
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-5">
          Recall by Drift Type
        </p>
        <div className="grid md:grid-cols-[1fr_280px] gap-6 items-start">
          <div className="bg-paper-50 border border-paper-300 rounded-xl p-6">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={byTypeData} margin={{ left: 4, right: 16, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8E5DF" vertical={false} />
                <XAxis dataKey="type" tick={{ fontSize: 11, fill: '#3D3D37' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6B6B63', fontFamily: 'JetBrains Mono' }} axisLine={false} />
                <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} contentStyle={{ backgroundColor: '#FDFCFA', border: '1px solid #E8E5DF', borderRadius: 8, fontSize: 12 }} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="CUSUM" fill="#2D7D6F" radius={[4, 4, 0, 0]} barSize={20} animationDuration={1000} animationBegin={0} animationEasing="ease-out" />
                <Bar dataKey="LSTM-AE" fill="#D4483B" radius={[4, 4, 0, 0]} barSize={20} animationDuration={1000} animationBegin={200} animationEasing="ease-out" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="space-y-3">
            <div className="p-4 rounded-xl bg-signal-teal-light border border-signal-teal/20">
              <p className="text-xs font-medium text-signal-teal mb-1">CUSUM Strengths</p>
              <p className="text-xs text-ink-200">Persistent drift: step (74%), multi-aspect (81%)</p>
            </div>
            <div className="p-4 rounded-xl bg-signal-red-light border border-signal-red/20">
              <p className="text-xs font-medium text-signal-red mb-1">LSTM-AE Strengths</p>
              <p className="text-xs text-ink-200">Transient anomalies (50% vs 42%) and fastest delay (0.41m)</p>
            </div>
          </div>
        </div>
      </motion.section>

      {/* LSTM-AE Architecture */}
      <motion.section variants={stagger.item} className="mb-12">
        <hr className="rule mb-6" />
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-5">
          LSTM-AE Architecture
        </p>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="p-6 rounded-xl bg-paper-50 border border-paper-300">
            <div className="font-mono text-xs text-ink-100 leading-loose space-y-1">
              {[
                ['Input', `[${drift.lstm_ae.seq_len} x 18] (9 scores + 9 masks)`],
                ['Encoder', `LSTM (hidden=${drift.lstm_ae.hidden_dim})`],
                ['Latent', `${drift.lstm_ae.latent_dim}-dim bottleneck`],
                ['Decoder', 'LSTM (mirror) → Linear'],
                ['Loss', 'Masked MSE → Anomaly Score'],
              ].map(([label, value]) => (
                <div key={label} className="flex items-center gap-3">
                  <span className="text-ink-50 w-16 text-right">{label}</span>
                  <span className="text-ink-50">→</span>
                  <span className="px-3 py-1 bg-paper-100 border border-paper-300 rounded text-ink-200">{value}</span>
                </div>
              ))}
            </div>
          </div>
          <div className="p-6 rounded-xl bg-paper-50 border border-paper-300">
            <p className="text-xs font-medium text-ink-50 uppercase tracking-wide mb-4">Training Details</p>
            <div className="space-y-2.5">
              {[
                ['Training set', `${drift.n_normal_train} normal products`],
                ['Seed ensemble', `${drift.lstm_ae.n_seeds} seeds, best selected`],
                ['Threshold', `${drift.lstm_ae.threshold.toFixed(3)} (synthetic val)`],
                ['Anomaly rate (drift)', `${(drift.lstm_ae.anomaly_rate_drift * 100).toFixed(1)}%`],
                ['Anomaly rate (normal)', `${(drift.lstm_ae.anomaly_rate_normal * 100).toFixed(1)}%`],
                ['Drift products', `${drift.n_drift_products}`],
              ].map(([label, value]) => (
                <div key={label} className="flex items-baseline justify-between text-sm">
                  <span className="text-ink-100">{label}</span>
                  <span className="font-mono text-xs text-ink-300">{value}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </motion.section>

      {/* ABSA Validation */}
      <motion.section variants={stagger.item} className="mb-12">
        <hr className="rule mb-6" />
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-5">
          ABSA Pipeline Validation
        </p>
        <div className="grid md:grid-cols-[1fr_280px] gap-6 items-start">
          <div className="bg-paper-50 border border-paper-300 rounded-xl p-6">
            <ResponsiveContainer width="100%" height={340}>
              <BarChart data={perAspectData} layout="vertical" margin={{ left: 4, right: 16 }} barGap={2}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E8E5DF" horizontal={false} />
                <XAxis type="number" domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} tick={{ fontSize: 10, fill: '#6B6B63', fontFamily: 'JetBrains Mono' }} />
                <YAxis type="category" dataKey="aspect" tick={{ fontSize: 11, fill: '#3D3D37' }} width={90} axisLine={false} tickLine={false} />
                <Tooltip formatter={(v: number, name: string) => [`${(v * 100).toFixed(1)}%`, name === 'dl' ? 'PyABSA' : 'TF-IDF+LR']} contentStyle={{ backgroundColor: '#FDFCFA', border: '1px solid #E8E5DF', borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="dl" radius={[0, 4, 4, 0]} barSize={12} name="PyABSA" animationDuration={1000} animationBegin={0} animationEasing="ease-out">
                  {perAspectData.map(d => <Cell key={d.aspect} fill={ASPECT_COLORS[d.aspect] ?? '#6B8CA3'} />)}
                </Bar>
                <Bar dataKey="lr" radius={[0, 4, 4, 0]} barSize={12} fill="#D4D0C8" name="TF-IDF+LR" animationDuration={1000} animationBegin={300} animationEasing="ease-out" />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3">
            <div className="p-4 rounded-xl bg-paper-50 border border-paper-300">
              <p className="text-xs font-medium text-ink-50 uppercase tracking-wide mb-3">Summary</p>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-ink-100">PyABSA Macro F1</span><span className="font-mono text-ink-300">{(absa.cv_dl_macro_f1 * 100).toFixed(1)}%</span></div>
                <div className="flex justify-between"><span className="text-ink-100">TF-IDF+LR Macro F1</span><span className="font-mono text-ink-100">{(absa.cv_lr_macro_f1 * 100).toFixed(1)}%</span></div>
                <div className="flex justify-between"><span className="text-ink-100">DL Advantage</span><span className="font-mono text-signal-teal">+{((absa.cv_dl_macro_f1 - absa.cv_lr_macro_f1) * 100).toFixed(1)}pp</span></div>
                <div className="flex justify-between"><span className="text-ink-100">Triplets</span><span className="font-mono text-ink-300">{absa.cv_n_triplets}</span></div>
                <div className="flex justify-between"><span className="text-ink-100">CV Folds</span><span className="font-mono text-ink-300">{absa.cv_n_folds}</span></div>
              </div>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Caveats & Limitations */}
      <motion.section variants={stagger.item}>
        <hr className="rule mb-6" />
        <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-5">
          Caveats & Limitations
        </p>
        <div className="grid md:grid-cols-2 gap-4">
          {[
            { title: 'Sparse Months', desc: 'Shrinkage smoothing mitigates noise in low-volume months, but does not eliminate it. Months with <3 reviews are masked entirely.' },
            { title: 'Amazon Demographic Bias', desc: 'Reviews skew toward US consumers, English-language, and Amazon purchase patterns. Verified/unverified purchase mix affects signal quality.' },
            { title: 'Weakly Supervised Attribution', desc: 'No ground-truth aspect labels exist for drift events. Aspect deltas and reconstruction errors are heuristic, not causal.' },
            { title: 'Human-in-the-Loop Required', desc: 'All detections are advisory. Business action requires manual review of evidence, context, and severity.' },
            { title: 'Historical Data Only', desc: 'System analyses past review data (Amazon 2015-2020). Not a real-time monitoring stream. Production deployment would require ingestion pipeline.' },
            { title: 'LSTM-AE Threshold Sensitivity', desc: 'Anomaly threshold tuned on synthetic validation set. Real-world operating point may need recalibration per product category.' },
          ].map(c => (
            <div key={c.title} className="p-4 rounded-xl bg-paper-50 border border-paper-300">
              <p className="text-sm font-medium text-ink-200 mb-1">{c.title}</p>
              <p className="text-xs text-ink-100 leading-relaxed">{c.desc}</p>
            </div>
          ))}
        </div>
      </motion.section>
    </motion.div>
  )
}
