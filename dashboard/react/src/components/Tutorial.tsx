import { useState, useEffect, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSlideTimeline } from '../hooks/useSlideTimeline'

interface Props { onDismiss: () => void }

const TOTAL_SLIDES = 28

/* ── colour map (mirrors tailwind.config) ─────────────────────────── */
const AC: Record<string, string> = {
  battery: '#C47D5C', camera: '#B5728A', display: '#A89070',
  software: '#5B86AB', build: '#8B7FB5', performance: '#6B8CA3',
  connectivity: '#C4A94D', audio: '#7C9EB2', charging: '#8BAA6E',
}
const SIG = { red: '#D4483B', teal: '#2D7D6F' }
const DET = { cusum: '#6A7B8B', lstm: '#4A8C7F' }
const INK = { dim: '#9C9C94', med: '#6B6B63', deep: '#3D3D37', black: '#1B1B18' }
const PAP = { cream: '#F0ECE5', bone: '#E8E5DF', stone: '#D4D0C8' }

/* ── mock data ────────────────────────────────────────────────────── */
const PROFILES = {
  A: [
    { a: 'battery', s: 0.82 }, { a: 'camera', s: 0.75 }, { a: 'display', s: 0.88 },
    { a: 'software', s: -0.45 }, { a: 'build', s: 0.71 }, { a: 'performance', s: -0.32 },
  ],
  B: [
    { a: 'battery', s: -0.61 }, { a: 'camera', s: 0.22 }, { a: 'display', s: 0.35 },
    { a: 'software', s: 0.79 }, { a: 'build', s: -0.48 }, { a: 'performance', s: 0.85 },
  ],
}

const NOISE = [.12, -.08, .15, -.03, .1, -.14, .07, -.11, .05, -.09, .13, -.06, .08, -.12, .04, -.1, .11, -.07, .06, -.13, .09, -.05, .14, -.08]
const TIMELINE = (() => {
  const raw: number[] = [], smooth: number[] = []
  for (let i = 0; i < 24; i++) {
    const base = 0.6 - (i > 14 ? (i - 14) * 0.06 : 0)
    raw.push(base + Math.sin(i * 0.7) * 0.15 + NOISE[i])
    smooth.push(base + Math.sin(i * 0.7) * 0.05)
  }
  return { raw, smooth }
})()

const CUSUM_OBS = [.62, .58, .63, .57, .61, .59, .60, .56, .62, .58, .52, .47, .43, .39, .36, .32, .29, .26, .24, .22]
const CUSUM_VALS = (() => {
  const vals: number[] = []; let s = 0
  for (const v of CUSUM_OBS) { s = Math.max(0, s + (0.6 - v) - 0.02); vals.push(s) }
  return vals
})()
const CUSUM_THRESH = 0.5

const LSTM_OBS = [.62, .59, .63, .58, .61, .60, .58, .56, .62, .55, .48, .35, .28, .24, .38, .52, .57, .60, .59, .61]
const LSTM_RECON = [.15, .18, .12, .21, .16, .19, .14, .22, .17, .38, .72, 1.45, 2.41, 2.78, 1.85, .52, .24, .18, .15, .14]
const LSTM_THRESH = 2.197

/* Source: results/drift/metrics.json (Phase 3 benchmark, n=200 synthetic cases).
   r = recall (fraction of drifts caught), d = mean detection delay in months. */
const DETECTORS = [
  { m: 'CUSUM', r: 0.570, d: 1.105, c: DET.cusum, active: true },
  { m: 'LSTM-AE', r: 0.335, d: 0.687, c: DET.lstm, active: true },
  { m: 'PELT', r: 0.155, d: 0.645, c: PAP.stone, active: false },
  { m: 'Star Rating', r: 0.010, d: 1.500, c: PAP.stone, active: false },
]

const DELTAS = [
  { a: 'connectivity', d: -0.90 }, { a: 'battery', d: -0.57 },
  { a: 'audio', d: -0.48 }, { a: 'performance', d: -0.37 }, { a: 'display', d: -0.26 },
]

/* ── new slide data ──────────────────────────────────────────────── */
const FUNNEL = [
  { label: 'Total Products', value: '1,501', bar: 1.0 },
  { label: 'Dense (12+ months)', value: '338', bar: 0.225 },
  { label: 'Significant Drift', value: '111', bar: 0.074 },
  { label: 'Case Studies', value: '5', bar: 0.003 },
]

type EdaStage = { group: string; items: string[] }
const EDA_PHASES: Array<{ phase: string; color: string; stages: EdaStage[] }> = [
  {
    phase: 'Acquire', color: SIG.red, stages: [
      { group: 'Ingest', items: ['Amazon Reviews 2023', 'Smartphone subset', 'Metadata join'] },
    ]
  },
  {
    phase: 'Clean', color: DET.cusum, stages: [
      { group: 'Text clean', items: ['HTML decode', 'URL strip', 'Whitespace norm', 'Title + body concat'] },
      { group: 'Row filters', items: ['Title keyword', 'Accessory removal', 'Null / empty drop', '< 5 tokens'] },
      { group: 'Dedup & lang', items: ['(user, asin, text, ts)', 'Non-English removed'] },
      { group: 'Metadata', items: ['helpful_vote ≥ 0', 'Rating 1–5', 'Verified flag'] },
      { group: 'Surge', items: ['Product-adaptive flag', 'is_surge_day'] },
    ]
  },
  {
    phase: 'Analyse', color: SIG.teal, stages: [
      { group: 'Density', items: ['12+ active months', 'Avg ≥ 5 / mo', 'Max gap ≤ 6 mo'] },
      { group: 'Drift scoring', items: ['Mann–Whitney U', 'Rolling Δ ≥ 0.5', '|Cohen d| ≥ 0.5', 'BH–FDR'] },
      { group: 'Case select', items: ['Top-5 ASINs', 'Split coverage', 'Control ASIN'] },
    ]
  },
  {
    phase: 'Ship', color: DET.lstm, stages: [
      { group: 'QC plots', items: ['Rating dist', 'Text length', 'Coverage heatmap', 'Density scatter', 'Timelines'] },
      { group: 'Outputs', items: ['Clean parquet', 'Density CSV', 'Drift scores', 'Case ASINs'] },
    ]
  },
]

/* Nexus 6P (B019BX7F86), battery aspect: real monthly data Jan 2016–Dec 2017, normalized (rating-1)/4.
   Healthy through Sep 2016, early decay Nov-Dec 2016, cliff May 2017, rock bottom Sep 2017. */
const NEXUS6P_BATTERY = [
  0.83, 0.85, 0.81, 0.70, 0.76, 0.71, 0.78, 0.71,
  0.76, 0.71, 0.61, 0.49, 0.62, 0.43, 0.46, 0.43,
  0.17, 0.10, 0.11, 0.07, 0.02, 0.13, 0.06, 0.08,
]

/* Nexus 6P, build/hardware aspect: real monthly data Jan 2016–Dec 2017.
   Holds longer than battery, then catastrophic cliff Apr 2017 when bootloop recognised as systemic design flaw. */
const NEXUS6P_BUILD = [
  0.75, 0.90, 0.76, 0.76, 0.66, 0.65, 0.71, 0.70,
  0.68, 0.56, 0.53, 0.52, 0.73, 0.61, 0.57, 0.16,
  0.13, 0.08, 0.14, 0.03, 0.08, 0.04, 0.15, 0.08,
]

/* ── framer slide variants ────────────────────────────────────────── */
const slideVariants = {
  enter: (d: number) => ({ x: d > 0 ? 200 : -200, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit: (d: number) => ({ x: d < 0 ? 200 : -200, opacity: 0 }),
}

/* ── SVG helpers ──────────────────────────────────────────────────── */
function dataPath(data: number[], x0: number, x1: number, y0: number, y1: number, vMin: number, vMax: number): string {
  return data.map((v, i) => {
    const x = x0 + (i / (data.length - 1)) * (x1 - x0)
    const y = y0 + (1 - (v - vMin) / (vMax - vMin)) * (y1 - y0)
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

/* ── shared layout pieces ─────────────────────────────────────────── */
function SlideLabel({ num, label }: { num: string; label: string }) {
  return (
    <div className="gs-label flex items-center gap-3 mb-4">
      <span className="font-mono text-xs text-signal-red font-semibold">{num}</span>
      <span className="font-mono text-xs uppercase tracking-[0.2em] text-ink-50">{label}</span>
    </div>
  )
}

function StatLine({ children, className }: { children: React.ReactNode; className?: string }) {
  return (
    <div className={`gs-stat flex items-baseline gap-3 mt-auto pt-4 border-t border-paper-300 ${className ?? ''}`}>
      {children}
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE BRIEF: Consultancy framing (i.e., client, blind spot, mandate)
   ════════════════════════════════════════════════════════════════════ */
function SlideBrief() {
  const containerRef = useSlideTimeline((tl) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-card', { autoAlpha: 0, y: 20, stagger: 0.15, duration: 0.5 }, 0.45)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.2')
  })

  const CARDS = [
    {
      tag: 'Client',
      accent: INK.deep,
      title: 'IntelliSys Ltd.',
      body: (
        <>
          Intelligent-systems integrator. Healthcare, finance, mobility, agriculture,{' '}
          digital media &amp; language
          Deep domain expertise, but no in-house deep-learning capability.
        </>
      ),
    },
    {
      tag: 'The Blind Spot',
      accent: SIG.red,
      title: 'Consumer electronics (smartphones)',
      body: (
        <>
          Quality issues surface <span className="font-medium text-signal-red">too late</span>, i.e.,
          after return spikes, review collapses, and social-media blowups.
          Star-rating dashboards only move once the damage is done.
        </>
      ),
    },
    {
      tag: 'The Mandate',
      accent: SIG.teal,
      title: 'A deep-learning early-warning system',
      body: (
        <>
          Read <span className="italic">what</span> is changing (battery? camera? software?){' '}
          <span className="italic">when</span> it starts, directly from review text,
          so IntelliSys can act before losses compound.
        </>
      ),
    },
  ]

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="02" label="The Brief" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Business Understanding
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-6">
        IntelliSys Ltd. asked us to detect product sentiment shift <span className="italic">before</span> it hits the bottom line.
      </p>

      <div className="grid grid-cols-3 gap-5 flex-1 min-h-0">
        {CARDS.map((c) => (
          <div
            key={c.tag}
            className="gs-card relative p-5 rounded-xl bg-paper-100 border border-paper-300 flex flex-col"
          >
            <span className="font-mono text-xs uppercase tracking-[0.2em] text-ink-50 mb-4">
              {c.tag}
            </span>
            <h3 className="font-display text-2xl text-ink-300 leading-snug mb-4">
              {c.title}
            </h3>
            <p className="text-base text-ink-100 leading-relaxed">
              {c.body}
            </p>
          </div>
        ))}
      </div>

      <StatLine className="!border-t-0">
        <span className="font-mono text-sm uppercase tracking-[0.2em] text-ink-50">Reactive</span>
        <svg className="text-ink-30" width="28" height="14" viewBox="0 0 28 14" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 7h24M19 1l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="font-mono text-sm uppercase tracking-[0.2em] text-signal-teal font-semibold">Proactive</span>
        <span className="text-sm text-ink-100 ml-3 italic">
          From reviews already written to signals acted on.
        </span>
      </StatLine>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 0: The Problem
   ════════════════════════════════════════════════════════════════════ */
function Slide0() {
  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-card', { autoAlpha: 0, y: 20, stagger: 0.12, duration: 0.5 }, 0.4)
      .from('.gs-rating', { autoAlpha: 0, scale: 0.8, stagger: 0.1, duration: 0.4 }, 0.6)
      .from('.gs-bar', { scaleX: 0, stagger: 0.04, duration: 0.5 }, 0.9)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.2')
  })

  const renderBars = (data: typeof PROFILES.A) => (
    <div className="space-y-2 mt-3">
      {data.map(({ a, s }) => (
        <div key={a} className="flex items-center gap-3">
          <span className="text-sm text-ink-100 w-24 capitalize">{a}</span>
          <div className="flex-1 flex items-center h-3 bg-paper-200 rounded-full relative">
            <div className="absolute inset-y-0 left-1/2 w-px bg-paper-400" />
            <div
              className="gs-bar h-full rounded-full absolute"
              style={{
                backgroundColor: s > 0 ? SIG.teal : SIG.red,
                left: s > 0 ? '50%' : `${50 + s * 50}%`,
                width: `${Math.abs(s) * 50}%`,
                transformOrigin: s > 0 ? 'left' : 'right',
              }}
            />
          </div>
          <span className={`font-mono text-sm w-12 text-right ${s > 0 ? 'text-signal-teal' : 'text-signal-red'}`}>
            {s > 0 ? '+' : ''}{s.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="03" label="The Problem" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Same rating. <span className="italic text-ink-100">Different products.</span>
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-6">
        Star ratings caught only 1.0% of drift events. Aspect-level analysis is essential.
      </p>

      <div className="grid grid-cols-2 gap-6 flex-1 min-h-0">
        {(['A', 'B'] as const).map(key => (
          <div key={key} className="gs-card p-5 rounded-xl bg-paper-100 border border-paper-300">
            <div className="flex items-baseline justify-between mb-1">
              <span className="text-base font-medium text-ink-200">Phone {key}</span>
              <span className="gs-rating font-mono text-2xl text-ink-300">3.8 ★</span>
            </div>
            <p className="text-sm text-ink-50 mb-2">
              {key === 'A' ? 'Great hardware, broken software' : 'Great software, degraded hardware'}
            </p>
            {renderBars(PROFILES[key])}
          </div>
        ))}
      </div>

      <div className="flex items-baseline justify-center gap-6 mt-auto pt-4">
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-3xl font-semibold text-signal-red">1.0%</span>
          <span className="text-base text-ink-50">with star ratings alone</span>
        </div>
        <svg className="text-ink-30" width="28" height="16" viewBox="0 0 28 16" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 8h24M19 1l6 7-6 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-3xl font-semibold text-signal-teal">57.0%</span>
          <span className="text-base text-ink-50">with aspect-level analysis</span>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 1: ABSA
   ════════════════════════════════════════════════════════════════════ */
function Slide1() {
  const TOKENS = [
    { word: 'battery', aspect: 'battery', score: -0.92 },
    { word: 'camera', aspect: 'camera', score: 0.85 },
    { word: 'Software', aspect: 'software', score: -0.78 },
  ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-review', { autoAlpha: 0, y: 16, duration: 0.5 }, 0.5)
      .from('.gs-highlight', { backgroundColor: 'rgba(0,0,0,0)', stagger: 0.25, duration: 0.4 }, 0.9)
      .from('.gs-pill', { autoAlpha: 0, y: 12, stagger: 0.2, duration: 0.35 }, 1.2)
      .from('.gs-pick', { autoAlpha: 0, y: 12, duration: 0.4 }, '>+0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <div>
        <SlideLabel num="06" label="ABSA" />
        <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
          Aspect-Based <span className="italic text-ink-100">Sentiment Analysis</span>
        </h2>
      </div>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-6">
        PyABSA<span className="font-mono text-[11px] font-semibold text-ink-50 align-super ml-0.5">[2]</span> extracts aspect-level sentiment from raw text. No fine-tuning needed.
      </p>

      <div className="gs-review flex-1 min-h-0 overflow-hidden bg-paper-100 border border-paper-300 rounded-xl p-5 flex flex-col justify-center">
        <p className="text-lg md:text-xl text-ink-200 leading-relaxed mb-4">
          &ldquo;The{' '}
          <span className="gs-highlight px-1.5 py-0.5 rounded-md" style={{ backgroundColor: `${AC.battery}30` }}>
            <span style={{ color: AC.battery }} className="font-medium">battery</span>
          </span>{' '}
          dies after 2 hours, but the{' '}
          <span className="gs-highlight px-1.5 py-0.5 rounded-md" style={{ backgroundColor: `${AC.camera}30` }}>
            <span style={{ color: AC.camera }} className="font-medium">camera</span>
          </span>{' '}
          is still excellent.{' '}
          <span className="gs-highlight px-1.5 py-0.5 rounded-md" style={{ backgroundColor: `${AC.software}30` }}>
            <span style={{ color: AC.software }} className="font-medium">Software</span>
          </span>{' '}
          update made everything slower.&rdquo;
        </p>

        <div className="flex flex-wrap gap-4 justify-center">
          {TOKENS.map(t => (
            <div key={t.aspect} className="gs-pill flex items-center gap-2 bg-white px-4 py-2 rounded-full border border-paper-300">
              <span className="w-3 h-3 rounded-full" style={{ backgroundColor: AC[t.aspect] }} />
              <span className="text-base font-medium capitalize text-ink-200">{t.aspect}</span>
              <span className={`font-mono text-base font-semibold ${t.score > 0 ? 'text-signal-teal' : 'text-signal-red'}`}>
                {t.score > 0 ? '+' : ''}{t.score.toFixed(2)}
              </span>
            </div>
          ))}
        </div>

        {/* Sentiment scale */}
        <div className="mt-5 flex flex-col items-center gap-1">
          <div className="flex items-center gap-2 w-64">
            <span className="font-mono text-[10px] text-signal-red">-1.0</span>
            <div className="flex-1 h-1.5 rounded-full" style={{
              background: `linear-gradient(to right, ${SIG.red}, ${PAP.bone} 50%, ${SIG.teal})`,
            }} />
            <span className="font-mono text-[10px] text-signal-teal">+1.0</span>
          </div>
          <span className="text-[10px] text-ink-50 tracking-wide">Negative &larr; Sentiment &rarr; Positive</span>
        </div>

      </div>

      {/* ── Before / After comparison ── */}
      <div className="gs-pick mt-3">
        <div className="grid grid-cols-2 gap-4 max-w-lg mx-auto">
          <div className="bg-paper-200 border border-paper-300 rounded-lg p-4 text-center">
            <span className="text-[10px] font-mono text-ink-50 uppercase tracking-wider block mb-2">Traditional</span>
            <span className="text-2xl tracking-wide">&#9733;&#9733;&#9733;&#9734;&#9734;</span>
            <p className="text-sm text-ink-100 mt-1 font-medium">&ldquo;Mixed feelings&rdquo;</p>
            <p className="text-[11px] text-ink-50 mt-2 italic">One number hides three stories.</p>
          </div>
          <div className="bg-paper-100 border border-paper-300 rounded-lg p-4" style={{ borderColor: `${SIG.teal}40` }}>
            <span className="text-[10px] font-mono text-signal-teal uppercase tracking-wider block mb-2 text-center">Aspect-Based</span>
            <div className="space-y-1.5">
              {TOKENS.map(t => (
                <div key={t.aspect} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: AC[t.aspect] }} />
                    <span className="text-sm capitalize text-ink-200">{t.aspect}</span>
                  </div>
                  <span className={`font-mono text-sm font-semibold ${t.score > 0 ? 'text-signal-teal' : 'text-signal-red'}`}>
                    {t.score > 0 ? '+' : ''}{t.score.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
            <p className="text-[11px] text-ink-50 mt-2 italic text-center">Actionable. Aspect by aspect.</p>
          </div>
        </div>
      </div>

    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 2: Feature Construction
   ════════════════════════════════════════════════════════════════════ */
function Slide2() {
  const W = 560, H = 180
  const X0 = 50, X1 = 540, Y0 = 15, Y1 = 155
  const vMin = -0.1, vMax = 1.0

  const rawPath = dataPath(TIMELINE.raw, X0, X1, Y0, Y1, vMin, vMax)
  const smoothPath = dataPath(TIMELINE.smooth, X0, X1, Y0, Y1, vMin, vMax)
  const rawDots = TIMELINE.raw.map((v, i) => ({
    cx: X0 + (i / 23) * (X1 - X0),
    cy: Y0 + (1 - (v - vMin) / (vMax - vMin)) * (Y1 - Y0),
  }))

  const containerRef = useSlideTimeline((tl, el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-chart', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.45)
      .from('.gs-dot', { autoAlpha: 0, scale: 0, stagger: 0.03, duration: 0.25 }, 0.6)

    // Draw the smoothed line
    const line = el.querySelector('.gs-smooth-line') as SVGPathElement | null
    if (line) {
      const len = line.getTotalLength()
      tl.fromTo('.gs-smooth-line',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.5, ease: 'power2.inOut' },
        1.0,
      )
    }

    tl.from('.gs-stats-overlay', { autoAlpha: 0, y: -8, duration: 0.4 }, '>-0.3')
      .from('.gs-explain-col', { autoAlpha: 0, y: 12, stagger: 0.1, duration: 0.4 }, '>-0.2')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="09" label="Feature Construction" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        From reviews to <span className="italic text-ink-100">time series</span>
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-5">
        Monthly aggregation with shrinkage smoothing and MAD normalisation
        <span className="text-base text-ink-50">, i.e., sparse months get pulled toward the product&rsquo;s long-run mean; scale uses robust spread, not z-score.</span>
      </p>

      <div className="gs-chart relative flex-1 bg-paper-100 border border-paper-300 rounded-xl p-4 overflow-hidden">
        <div className="gs-stats-overlay absolute top-3 right-3 text-right text-xs text-ink-50 leading-relaxed">
          <div><span className="font-mono font-semibold text-ink-300">335</span> products &middot; <span className="font-mono font-semibold text-ink-300">24+</span> months</div>
          <div><span className="font-mono font-semibold text-ink-300">6&times;18</span> window &middot; shrinkage + MAD</div>
        </div>
        <div className="flex gap-4 mb-2 text-xs text-ink-50">
          <span className="flex items-center gap-2"><span className="w-4 h-px inline-block" style={{ backgroundColor: PAP.stone }} /> Raw</span>
          <span className="flex items-center gap-2"><span className="w-4 h-0.5 inline-block rounded" style={{ backgroundColor: SIG.teal }} /> Smoothed</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="overflow-visible">
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1.0].map(v => {
            const y = Y0 + (1 - (v - vMin) / (vMax - vMin)) * (Y1 - Y0)
            return <line key={v} x1={X0} x2={X1} y1={y} y2={y} stroke={PAP.bone} strokeDasharray="3 3" />
          })}
          {/* Y-axis labels */}
          {[0, 0.5, 1.0].map(v => {
            const y = Y0 + (1 - (v - vMin) / (vMax - vMin)) * (Y1 - Y0)
            return <text key={v} x={X0 - 8} y={y + 3} textAnchor="end" fontSize="9" fill={INK.dim} fontFamily="JetBrains Mono">{v.toFixed(1)}</text>
          })}
          {/* X-axis labels */}
          {[0, 6, 12, 18, 23].map(i => (
            <text key={i} x={X0 + (i / 23) * (X1 - X0)} y={Y1 + 14} textAnchor="middle" fontSize="8" fill={INK.dim} fontFamily="JetBrains Mono">
              {`${2015 + Math.floor(i / 12)}-${String((i % 12) + 1).padStart(2, '0')}`}
            </text>
          ))}
          {/* Raw dots */}
          {rawDots.map((d, i) => (
            <circle key={i} className="gs-dot" cx={d.cx} cy={d.cy} r={2.5} fill={PAP.stone} />
          ))}
          {/* Raw path (faint) */}
          <path d={rawPath} fill="none" stroke={PAP.stone} strokeWidth={1} opacity={0.5} />
          {/* Smoothed line (GSAP draws this) */}
          <path className="gs-smooth-line" d={smoothPath} fill="none" stroke={SIG.teal} strokeWidth={2.5} strokeLinecap="round" />
        </svg>
      </div>

      <div className="mt-5 grid grid-cols-2 gap-8 text-sm">
        <div className="gs-explain-col">
          <div className="flex items-baseline gap-2 mb-2">
            <span className="font-display text-base font-semibold text-ink-300">Shrinkage</span>
            <span className="font-mono text-[11px] text-ink-50 uppercase tracking-wider">empirical Bayes</span>
          </div>
          <ul className="space-y-1.5 text-ink-100 leading-snug">
            <li className="flex gap-2"><span className="text-signal-teal">&bull;</span><span>Pulls sparse months toward the product&rsquo;s long-run mean</span></li>
            <li className="flex gap-2"><span className="text-signal-teal">&bull;</span><span>Weight scales with monthly review count, i.e., noisy months trust the prior</span></li>
            <li className="flex gap-2"><span className="text-signal-teal">&bull;</span><span>Prevents 3-review months from looking like real drift</span></li>
          </ul>
        </div>
        <div className="gs-explain-col">
          <div className="flex items-baseline gap-2 mb-2">
            <span className="font-display text-base font-semibold text-ink-300">MAD</span>
            <span className="font-mono text-[11px] text-ink-50 uppercase tracking-wider">median abs. deviation</span>
          </div>
          <ul className="space-y-1.5 text-ink-100 leading-snug">
            <li className="flex gap-2"><span className="text-signal-teal">&bull;</span><span>Robust scale estimate, i.e., unaffected by outlier months</span></li>
            <li className="flex gap-2"><span className="text-signal-teal">&bull;</span><span>Normalises every product to a common unit for cross-product training</span></li>
            <li className="flex gap-2"><span className="text-signal-teal">&bull;</span><span>Lets one global LSTM-AE learn a shared drift signature</span></li>
          </ul>
        </div>
      </div>

    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 3: LSTM-AE Architecture
   ════════════════════════════════════════════════════════════════════ */
function Slide3() {
  const CY = 100
  const nodeData = [
    { cx: 60, w: 100, h: 90, label: 'Input', sub: '6 × 18', fill: PAP.cream, stroke: PAP.bone, sw: 1.5 },
    { cx: 210, w: 120, h: 66, label: 'Encoder LSTM', sub: 'hidden = 32', fill: `${SIG.teal}0A`, stroke: `${SIG.teal}28`, sw: 1.5 },
    { cx: 370, w: 90, h: 42, label: 'Latent Space', sub: 'dim = 8', fill: `${SIG.teal}18`, stroke: SIG.teal, sw: 2 },
    { cx: 530, w: 120, h: 66, label: 'Decoder LSTM', sub: 'hidden = 32', fill: `${SIG.teal}0A`, stroke: `${SIG.teal}28`, sw: 1.5 },
    { cx: 680, w: 130, h: 90, label: 'Reconstruction', sub: '6 × 9', fill: PAP.cream, stroke: PAP.bone, sw: 1.5 },
  ]

  /* Tapered flow polygons between adjacent nodes */
  const flows = nodeData.slice(0, -1).map((from, i) => {
    const to = nodeData[i + 1]
    const x1 = from.cx + from.w / 2, x2 = to.cx - to.w / 2
    const ft = CY - from.h / 2, fb = CY + from.h / 2
    const tt = CY - to.h / 2, tb = CY + to.h / 2
    return `${x1},${ft} ${x2},${tt} ${x2},${tb} ${x1},${fb}`
  })

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-arch', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.45)
      // Flow connectors taper in
      .from('.gs-flow', { autoAlpha: 0, stagger: 0.08, duration: 0.3 }, 0.55)
      // Nodes scale from their own centres
      .from('.gs-node', { autoAlpha: 0, scale: 0.85, transformOrigin: 'center center', stagger: 0.1, duration: 0.45 }, 0.6)
      // Dimension annotations
      .from('.gs-dim', { autoAlpha: 0, y: 5, stagger: 0.05, duration: 0.25 }, 1.1)
      // Equation reveal
      .from('.gs-error', { autoAlpha: 0, y: 12, duration: 0.4 }, 1.3)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="10" label="LSTM-Autoencoder" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Long Short-Term Memory
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-5">
        Trained on drift-free products only. High reconstruction error signals drift.
        <span className="block text-sm text-ink-50 mt-1">MC-Dropout is on during training, off at validation, so val loss can sit below train loss without overfitting.</span>
      </p>

      <div className="gs-arch flex-1 bg-paper-100 border border-paper-300 rounded-xl p-4 flex flex-col justify-center overflow-hidden">
        <svg viewBox="0 0 740 200" width="100%" className="overflow-visible">
          {/* Flow connectors: tapered trapezoids showing dimensional compression */}
          {flows.map((pts, i) => (
            <polygon key={i} className="gs-flow" points={pts}
              fill={`${SIG.teal}06`} stroke={`${SIG.teal}10`} strokeWidth={0.5}
            />
          ))}
          {/* Architecture nodes: heights proportional to dimensionality */}
          {nodeData.map((n, i) => (
            <g key={i} className="gs-node" transform={`translate(${n.cx}, ${CY})`}>
              <rect x={-n.w / 2} y={-n.h / 2} width={n.w} height={n.h} rx={i === 2 ? 6 : 8}
                fill={n.fill} stroke={n.stroke} strokeWidth={n.sw}
              />
              <text x={0} y={i === 2 ? -2 : -6} textAnchor="middle"
                fontSize={i === 2 ? '13' : '14'} fontWeight="500" fill={INK.black}
              >{n.label}</text>
              <text x={0} y={i === 2 ? 14 : 12} textAnchor="middle"
                fontSize="11" fontFamily="JetBrains Mono" fill={INK.dim}
              >{n.sub}</text>
            </g>
          ))}
          {/* Dimensionality annotations: show the compression ratio */}
          {[
            { x: 60, y: CY - 51, t: '108', c: INK.dim, fw: '400' },
            { x: 210, y: CY - 39, t: '32', c: INK.dim, fw: '400' },
            { x: 370, y: CY - 24, t: '8', c: SIG.teal, fw: '600' },
            { x: 530, y: CY - 39, t: '32', c: INK.dim, fw: '400' },
            { x: 680, y: CY - 51, t: '108', c: INK.dim, fw: '400' },
          ].map((d, i) => (
            <text key={i} className="gs-dim" x={d.x} y={d.y} textAnchor="middle"
              fontSize="10" fontFamily="JetBrains Mono" fill={d.c} fontWeight={d.fw}
              opacity={d.fw === '600' ? 1 : 0.55}
            >{d.t}</text>
          ))}
        </svg>

        {/* Anomaly equation: editorial flanking-rule treatment */}
        <div className="gs-error mt-2">
          <p className="text-sm text-ink-50 text-center mt-1.5">
            Threshold: 2.197 &middot; 3-seed ensemble &middot; 224 drift-free training products &middot; input 6&times;18 = 9 aspect scores + 9 masks; reconstructs the 9 sentiments
          </p>
        </div>
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-auto pt-4 justify-center">
        <span className="font-mono text-4xl font-semibold text-ink-300">0.69</span>
        <span className="text-lg text-ink-50">month avg detection delay &middot; faster than CUSUM (1.11) when it fires</span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 4: LSTM-AE Detection
   ════════════════════════════════════════════════════════════════════ */
function SlideLSTMDetect() {
  const W = 560, H = 180
  const X0 = 50, X1 = 530, Y0 = 15, Y1 = 155
  const maxRecon = Math.max(...LSTM_RECON) * 1.15

  const obsPath = dataPath(LSTM_OBS, X0, X1, Y0, Y1, 0, 1)
  const reconPath = dataPath(LSTM_RECON, X0, X1, Y0, Y1, 0, maxRecon)
  const threshY = Y0 + (1 - LSTM_THRESH / maxRecon) * (Y1 - Y0)

  // Interpolate exact crossing point between the sample before and after threshold
  const crossIdxRaw = LSTM_RECON.findIndex(v => v >= LSTM_THRESH)
  const crossIdx = crossIdxRaw
  const crossFrac = (() => {
    if (crossIdxRaw <= 0) return crossIdxRaw
    const prev = LSTM_RECON[crossIdxRaw - 1], cur = LSTM_RECON[crossIdxRaw]
    return crossIdxRaw - 1 + (LSTM_THRESH - prev) / (cur - prev)
  })()
  const crossX = crossIdx >= 0 ? X0 + (crossFrac / (LSTM_RECON.length - 1)) * (X1 - X0) : 0
  const _peakReconY = Y0 + (1 - Math.max(...LSTM_RECON) / maxRecon) * (Y1 - Y0); void _peakReconY

  const containerRef = useSlideTimeline((tl, el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-chart', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.45)

    const obsLine = el.querySelector('.gs-obs-line') as SVGPathElement | null
    if (obsLine) {
      const len = obsLine.getTotalLength()
      tl.fromTo('.gs-obs-line',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.2, ease: 'power2.inOut' }, 0.6)
    }
    const reconLine = el.querySelector('.gs-recon-line') as SVGPathElement | null
    if (reconLine) {
      const len = reconLine.getTotalLength()
      tl.fromTo('.gs-recon-line',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.5, ease: 'power2.inOut' }, 0.8)
    }
    tl.from('.gs-thresh', { autoAlpha: 0, duration: 0.4 }, 1.5)
    if (crossIdx >= 0) {
      tl.from('.gs-cross', { autoAlpha: 0, scale: 0, duration: 0.4, svgOrigin: `${crossX} ${threshY}` }, 1.8)
        .to('.gs-cross-ring', { scale: 1.8, autoAlpha: 0, duration: 0.8, ease: 'power1.out', svgOrigin: `${crossX} ${threshY}` }, 2.0)
    }
    tl.from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="11" label="LSTM-AE Detection" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Reconstruction error <span className="italic text-ink-100">reveals drift</span>
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-5">
        When the autoencoder can&rsquo;t reconstruct what it learned as normal, something changed.
      </p>

      <div className="gs-chart flex-1 bg-paper-100 border border-paper-300 rounded-xl p-4 overflow-hidden">
        <div className="flex gap-4 mb-2 text-xs text-ink-50">
          <span className="flex items-center gap-2"><span className="w-4 h-0.5 inline-block rounded" style={{ backgroundColor: INK.deep }} /> Sentiment</span>
          <span className="flex items-center gap-2"><span className="w-4 h-0.5 inline-block rounded" style={{ backgroundColor: SIG.red }} /> Recon. Error</span>
          <span className="flex items-center gap-2"><span className="w-4 border-t border-dashed inline-block" style={{ borderColor: SIG.red }} /> Threshold</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="overflow-visible">
          {[0, 0.25, 0.5, 0.75, 1.0].map(v => {
            const y = Y0 + (1 - v) * (Y1 - Y0)
            return <line key={v} x1={X0} x2={X1} y1={y} y2={y} stroke={PAP.bone} strokeDasharray="3 3" />
          })}
          <line className="gs-thresh" x1={X0} x2={X1} y1={threshY} y2={threshY}
            stroke={SIG.red} strokeWidth={1.5} strokeDasharray="6 4" opacity={0.7}
          />
          <text className="gs-thresh" x={X1 + 4} y={threshY + 3} fontSize="9" fill={SIG.red} fontFamily="JetBrains Mono">&theta;</text>
          <path className="gs-obs-line" d={obsPath} fill="none" stroke={INK.deep} strokeWidth={1.5} strokeLinecap="round" />
          <path className="gs-recon-line" d={reconPath} fill="none" stroke={SIG.red} strokeWidth={2.5} strokeLinecap="round" />
          {crossIdx >= 0 && (
            <g className="gs-cross">
              <circle className="gs-cross-ring" cx={crossX} cy={threshY} r={8} fill="none" stroke={SIG.red} strokeWidth={2} />
              <circle cx={crossX} cy={threshY} r={4} fill={SIG.red} />
              <text x={crossX} y={Y0 + 0.125 * (Y1 - Y0)} textAnchor="middle" fontSize="10" fontWeight="600" fill={SIG.red} fontFamily="JetBrains Mono">
                DRIFT
              </text>
            </g>
          )}
          {[0, 5, 10, 15, 19].map(i => (
            <text key={i} x={X0 + (i / 19) * (X1 - X0)} y={Y1 + 14} textAnchor="middle" fontSize="8" fill={INK.dim} fontFamily="JetBrains Mono">
              t{i}
            </text>
          ))}
        </svg>
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-auto pt-4 justify-center">
        <span className="font-mono text-3xl font-semibold text-signal-red">33.5%</span>
        <span className="text-base text-ink-50">recall &middot; 0.69-month delay &middot; ~1.6&times; faster than CUSUM when it fires</span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 5: CUSUM Architecture
   ════════════════════════════════════════════════════════════════════ */
function SlideCUSUMArch() {
  /* Layout constants */
  const VW = 660, VH = 215
  const X0 = 52, X1 = 635, N = CUSUM_OBS.length

  /* Upper panel: observations (y 18–88) */
  const OY0 = 18, OY1 = 88
  const obsPath = dataPath(CUSUM_OBS, X0, X1, OY0, OY1, 0.15, 0.72)
  const baseY = OY0 + (1 - (0.6 - 0.15) / (0.72 - 0.15)) * (OY1 - OY0)

  /* Lower panel: CUSUM accumulation (y 112–195) */
  const CY0 = 112, CY1 = 195
  const maxC = Math.max(...CUSUM_VALS) * 1.15
  const cusLine = dataPath(CUSUM_VALS, X0, X1, CY0, CY1, 0, maxC)
  const cusArea = cusLine + ` L${X1},${CY1} L${X0},${CY1} Z`
  const threshY = CY0 + (1 - CUSUM_THRESH / maxC) * (CY1 - CY0)

  /* Crossing */
  const crossIdx = CUSUM_VALS.findIndex(v => v >= CUSUM_THRESH)
  const crossX = X0 + (crossIdx / (N - 1)) * (X1 - X0)
  const crossNorm = (crossX - X0) / (X1 - X0)

  /* Per-point data for dots + deviation arrows */
  const pts = CUSUM_OBS.map((v, i) => ({
    x: X0 + (i / (N - 1)) * (X1 - X0),
    y: OY0 + (1 - (v - 0.15) / (0.72 - 0.15)) * (OY1 - OY0),
    dev: 0.6 - v,
  }))

  const containerRef = useSlideTimeline((tl, el) => {
    /* header */
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 12, duration: 0.35 }, 0.3)
      .from('.gs-viz', { autoAlpha: 0, y: 14, duration: 0.4 }, 0.4)

    /* baseline */
    tl.from('.gs-base', { autoAlpha: 0, duration: 0.2 }, 0.5)

    /* observation line draw */
    const obsEl = el.querySelector('.gs-obs') as SVGPathElement | null
    if (obsEl) {
      const len = obsEl.getTotalLength()
      tl.fromTo('.gs-obs',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.4, ease: 'none' }, 0.55)
    }

    /* data dots stagger */
    tl.from('.gs-dot', {
      autoAlpha: 0, scale: 0, transformOrigin: 'center',
      stagger: 0.04, duration: 0.15,
    }, 0.6)

    /* deviation arrows */
    tl.from('.gs-dev', {
      autoAlpha: 0, scaleY: 0, transformOrigin: 'top center',
      stagger: 0.04, duration: 0.22,
    }, 1.0)

    /* threshold */
    tl.from('.gs-thresh', { autoAlpha: 0, duration: 0.25 }, 1.1)

    /* CUSUM area + line progressive reveal */
    tl.fromTo('.gs-cus-clip rect',
      { attr: { width: 0 } },
      { attr: { width: X1 - X0 + 10 }, duration: 1.5, ease: 'power1.inOut' }, 1.25)

    /* drift badge + ring pulse */
    tl.from('.gs-drift', {
      autoAlpha: 0, scale: 0.7, transformOrigin: 'center',
      duration: 0.4, ease: 'back.out(1.5)',
    }, 2.45)
    tl.to('.gs-cross-ring', {
      scale: 1.8, autoAlpha: 0, duration: 0.8, ease: 'power1.out',
      svgOrigin: `${crossX} ${threshY}`,
    }, 2.65)

    /* step labels */
    tl.from('.gs-step', {
      autoAlpha: 0, y: 5, stagger: 0.12, duration: 0.25,
    }, 0.65)

    /* formula + stat */
    tl.from('.gs-formula', { autoAlpha: 0, y: 10, duration: 0.35 }, 2.6)
    tl.from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="12" label="CUSUM" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-2">
        Cumulative Sum<span className="font-mono text-sm font-semibold text-ink-50 align-super ml-1">[5]</span>
      </h2>
      <p className="gs-caption text-sm text-ink-100 leading-relaxed mb-4">
        Deviations accumulate until threshold is crossed — two-sided, no training required.
      </p>

      <div className="gs-viz flex-1 bg-paper-100 border border-paper-300 rounded-xl p-4 overflow-hidden">
        <div className="flex gap-4 mb-2 text-xs text-ink-50">
          <span className="flex items-center gap-2"><span className="w-4 h-0.5 inline-block rounded" style={{ backgroundColor: INK.med }} /> Observation</span>
          <span className="flex items-center gap-2"><span className="w-4 h-0.5 inline-block rounded" style={{ backgroundColor: SIG.teal }} /> CUSUM S⁻</span>
          <span className="flex items-center gap-2"><span className="w-4 border-t border-dashed inline-block" style={{ borderColor: INK.dim }} /> Threshold h</span>
        </div>
        <svg viewBox={`0 0 ${VW} ${VH}`} width="100%" className="overflow-visible">
          <defs>
            <clipPath id="gs-cus-clip" className="gs-cus-clip">
              <rect x={X0} y={0} width={0} height={VH} />
            </clipPath>
            <linearGradient id="gs-cus-grad"
              gradientUnits="userSpaceOnUse" x1={X0} x2={X1} y1={0} y2={0}>
              <stop offset={Math.max(0, crossNorm - 0.02)} stopColor={SIG.teal} />
              <stop offset={Math.min(1, crossNorm + 0.02)} stopColor={SIG.red} />
            </linearGradient>
          </defs>

          {/* ── Upper: Observations ── */}
          <line className="gs-base" x1={X0} x2={X1} y1={baseY} y2={baseY}
            stroke={INK.dim} strokeDasharray="4 3" strokeWidth={0.8} opacity={0.4} />
          <text className="gs-base" x={X0 - 7} y={baseY + 3.5} textAnchor="end"
            fontSize="8" fontFamily="JetBrains Mono" fill={INK.dim} opacity={0.55}>&mu;&#x2080;</text>

          <path className="gs-obs" d={obsPath} fill="none"
            stroke={INK.med} strokeWidth={1.8} strokeLinecap="round" opacity={0.6} />

          {pts.map((p, i) => (
            <circle key={i} className="gs-dot" cx={p.x} cy={p.y} r={2.2}
              fill={p.dev > 0.05 ? SIG.red : INK.dim}
              opacity={p.dev > 0.05 ? 0.55 : 0.25} />
          ))}

          {pts.filter(p => p.dev > 0.05).map((p, i) => (
            <line key={i} className="gs-dev"
              x1={p.x} y1={baseY} x2={p.x} y2={p.y}
              stroke={SIG.red} strokeWidth={1} opacity={0.22} />
          ))}

          {/* separator */}
          <line x1={X0} x2={X1} y1={100} y2={100}
            stroke={PAP.bone} strokeWidth={0.5} opacity={0.5} />

          {/* ── Lower: CUSUM Accumulation ── */}
          <line className="gs-thresh" x1={X0} x2={X1} y1={threshY} y2={threshY}
            stroke={INK.dim} strokeDasharray="5 4" strokeWidth={0.8} />
          <text className="gs-thresh" x={X0 - 7} y={threshY + 3.5} textAnchor="end"
            fontSize="8" fontFamily="JetBrains Mono" fill={INK.dim}>h</text>

          <g clipPath="url(#gs-cus-clip)">
            <path d={cusArea} fill="url(#gs-cus-grad)" fillOpacity={0.09} />
            <path d={cusLine} fill="none"
              stroke="url(#gs-cus-grad)" strokeWidth={2.5} strokeLinecap="round" />
          </g>

          {crossIdx >= 0 && (
            <g className="gs-drift">
              <circle className="gs-cross-ring" cx={crossX} cy={threshY} r={8} fill="none" stroke={SIG.red} strokeWidth={2} />
              <circle cx={crossX} cy={threshY} r={4} fill={SIG.red} />
              <text x={crossX} y={threshY - 14} textAnchor="middle" fill={SIG.red}
                fontSize={8} fontFamily="JetBrains Mono" fontWeight={600}
                letterSpacing="0.06em">DRIFT</text>
            </g>
          )}

          {/* step labels positioned at meaningful locations */}
          {[
            { x: X0 + 28, y: OY1 + 5, t: 'OBSERVE', c: INK.dim },
            { x: pts[13]?.x ?? 430, y: OY0 - 4, t: 'DEVIATE', c: SIG.red },
            { x: X0 + (X1 - X0) * 0.35, y: CY1 + 12, t: 'ACCUMULATE', c: SIG.teal },
            { x: Math.min(crossX + 60, X1 - 25), y: threshY + 18, t: 'SIGNAL', c: SIG.red },
          ].map((s, i) => (
            <text key={i} className="gs-step" x={s.x} y={s.y}
              textAnchor="middle" fontSize="7" fontFamily="JetBrains Mono"
              fill={s.c} letterSpacing="0.12em" fontWeight="500" opacity={0.65}>{s.t}</text>
          ))}

          {/* x-axis time labels */}
          {[0, 5, 10, 15, 19].map(i => (
            <text key={i} x={X0 + (i / 19) * (X1 - X0)} y={VH + 10}
              textAnchor="middle" fontSize="7" fill={INK.dim} fontFamily="JetBrains Mono">
              t{i}
            </text>
          ))}
        </svg>
      </div>

      <div className="gs-formula mt-3 text-center">
        <p className="font-mono text-[15px] text-ink-200">
          <span className="font-semibold" style={{ color: SIG.teal }}>S⁻ₜ</span>
          <span className="text-ink-50"> = </span>max(0, S⁻ₜ₋₁ + (&mu;&#x2080; &minus; x&#x209C;) &minus; k)
          <span className="text-ink-50 mx-1">,</span>
          alert if S⁻ₜ &ge; h
          <span className="text-ink-50 mx-2">&middot;</span>
          &mu;&#x2080; = 0.60
          <span className="text-ink-50 mx-2">&middot;</span>
          k = 0.02
          <span className="text-ink-50 mx-2">&middot;</span>
          h = 0.50
        </p>
        <p className="font-mono text-[10px] text-ink-50/80 mt-1">
          illustrative one-sided decline form &middot; production detector is two-sided on z-scored norm_scores with k = 0.5
        </p>
      </div>

      <div className="gs-stat flex items-baseline justify-center gap-3 mt-auto pt-4 flex-wrap">
        <span className="font-mono text-3xl font-semibold text-ink-300">0</span>
        <span className="text-base text-ink-50">training data needed</span>
        <span className="text-ink-50 text-sm mx-1">&middot;</span>
        <span className="font-mono text-3xl font-semibold" style={{ color: SIG.teal }}>57.0%</span>
        <span className="text-base text-ink-50">recall &middot; best on persistent drift (73.1%)</span>
      </div>
    </div>
  )
}


/* ════════════════════════════════════════════════════════════════════
   SLIDE 12: DL vs Classical (critical reflection)
   CUSUM dominates recall on every drift type. LSTM-AE kept as the
   DL benchmark; its only edge is ~0.4 mo faster firing conditional
   on a hit, i.e., partly selection bias (catches easier drifts only).
   ════════════════════════════════════════════════════════════════════ */
function Slide5() {
  const maxR = 0.7
  const barW = 80

  const maxDelay = 1.2
  const active = DETECTORS.filter(d => d.active)

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-bars', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.45)
      .from('.gs-bar-fill', { scaleY: 0, stagger: 0.12, duration: 0.6 }, 0.6)
      .to('.gs-faded', { opacity: 0.3, duration: 0.4 }, 1.4)
      .from('.gs-latency', { autoAlpha: 0, y: 16, duration: 0.4 }, 1.0)
      .from('.gs-lat-bar', { scaleX: 0, stagger: 0.15, duration: 0.7 }, 1.2)
      .from('.gs-lat-label', { autoAlpha: 0, x: -8, stagger: 0.15, duration: 0.3 }, 1.6)
      .from('.gs-alert', { autoAlpha: 0, y: 10, stagger: 0.1, duration: 0.35 }, 2.0)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0 items-center text-center">
      <SlideLabel num="13" label="DL vs Classical" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Classical <span className="italic text-ink-100">vs</span> Deep Learning
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-5">
        CUSUM wins recall on every drift type (57.0% vs 33.5% overall). LSTM-AE&rsquo;s only edge: firing ~0.4 mo earlier when it hits.
      </p>

      <div className="grid grid-cols-2 gap-5 flex-1 min-h-0 items-start w-full">
        {/* Recall bar chart */}
        <div className="gs-bars bg-paper-100 border border-paper-300 rounded-xl p-5 flex flex-col">
          <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-4">Recall</p>
          <div className="flex items-end justify-center gap-6 flex-1 min-h-0">
            {DETECTORS.map(d => {
              const h = (d.r / maxR) * 150
              return (
                <div key={d.m} className={`flex flex-col items-center ${d.active ? '' : 'gs-faded'}`}>
                  <span className="font-mono text-sm font-semibold text-ink-300 mb-1">{(d.r * 100).toFixed(1)}%</span>
                  <div className="gs-bar-fill rounded-t-md" style={{
                    width: barW, height: h, backgroundColor: d.c,
                    transformOrigin: 'bottom',
                    opacity: d.active ? 1 : 0.3,
                  }} />
                  <span className="text-xs text-ink-200 mt-2 text-center font-medium">{d.m}</span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Detection latency comparison */}
        <div className="gs-latency bg-paper-100 border border-paper-300 rounded-xl p-5 flex flex-col">
          <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-4">
            Detection Latency <span className="normal-case tracking-normal italic text-ink-50">(when detected)</span>
          </p>
          <div className="flex flex-col gap-5 flex-1 justify-center">
            {active.map(d => {
              const pct = (d.d / maxDelay) * 100
              return (
                <div key={d.m} className="space-y-1.5">
                  <div className="flex items-baseline justify-between">
                    <span className="font-mono text-xs font-medium text-ink-200">{d.m}</span>
                    <span className="gs-lat-label font-mono text-sm font-semibold text-ink-300">{d.d.toFixed(2)}m</span>
                  </div>
                  <div className="relative h-7 bg-paper-200/60 rounded-md overflow-hidden">
                    <div className="gs-lat-bar absolute inset-y-0 left-0 rounded-md" style={{
                      width: `${pct}%`, backgroundColor: d.c, opacity: 0.85,
                      transformOrigin: 'left',
                    }} />
                  </div>
                </div>
              )
            })}
            {/* Selection-bias caveat on the speed edge */}
            <div className="flex items-center gap-2 pt-2 border-t border-paper-300">
              <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="shrink-0">
                <circle cx="8" cy="8" r="6.5" stroke={INK.med} strokeWidth="1.2" opacity="0.7" />
                <circle cx="8" cy="5" r="0.7" fill={INK.med} opacity="0.7" />
                <path d="M8 7.5v3.5" stroke={INK.med} strokeWidth="1.2" strokeLinecap="round" opacity="0.7" />
              </svg>
              <p className="font-mono text-xs text-ink-200 text-left">
                Speed edge is partly <span className="font-semibold text-ink-300">selection bias</span>, i.e., LSTM-AE only catches the easier drifts.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Why classical won: critical reflection strip */}
      <div className="gs-alert mt-4 p-4 rounded-xl bg-paper-200/60 border border-paper-300 flex items-center justify-center gap-6 flex-wrap w-full">
        <span className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50">Why classical won</span>
        <span className="flex items-center gap-2 text-sm">
          <span className="w-2 h-2 rounded-full bg-ink-300" />
          <span className="text-ink-100">Short windows &rarr;</span>
          <span className="font-mono font-semibold text-ink-200">seq_len = 6</span>
        </span>
        <span className="flex items-center gap-2 text-sm">
          <span className="w-2 h-2 rounded-full bg-ink-300" />
          <span className="text-ink-100">Sparse signal &rarr;</span>
          <span className="font-mono font-semibold text-ink-200">9-dim reconstruction</span>
        </span>
        <span className="flex items-center gap-2 text-sm">
          <span className="w-2 h-2 rounded-full bg-ink-300" />
          <span className="text-ink-100">Matched baseline &rarr;</span>
          <span className="font-mono font-semibold text-ink-200">CUSUM fits cumulative drift</span>
        </span>
      </div>

      <StatLine>
        <span className="text-base text-ink-50">
          DL isn&rsquo;t always the right tool. Classical monitoring beats DL on short, sparse multivariate sequences.
        </span>
      </StatLine>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 7: Explainability
   ════════════════════════════════════════════════════════════════════ */
/* ════════════════════════════════════════════════════════════════════
   SLIDE: Explanation Primitives (bridge Phase 3 → Phase 4)
   Introduces the four tools we use to turn a detected drift into a story.
   ════════════════════════════════════════════════════════════════════ */
function SlideExplainPrimitives() {
  const CARDS = [
    {
      num: '01',
      title: 'Aspect delta',
      caption: 'What changed',
      body: 'Raw before/after sentiment per aspect. Flagged high-confidence when both windows have ≥2 observations.',
      accent: SIG.red,
    },
    {
      num: '02',
      title: 'Reconstruction error',
      caption: 'Where in latent space',
      body: 'Per-aspect LSTM-AE error decomposition, i.e., which aspect the deep detector found anomalous in its learned manifold.',
      accent: DET.lstm,
    },
    {
      num: '03',
      title: 'Detector agreement',
      caption: 'Do the methods converge',
      body: 'Overlap between CUSUM\u2019s aspect ranking and the LSTM-AE\u2019s top contributors, i.e., consensus or complementary signal.',
      accent: DET.cusum,
    },
    {
      num: '04',
      title: 'Review evidence',
      caption: 'Customer language',
      body: 'Representative quotes drawn from the drift window, aspect-filtered through PyABSA [2]. Analyst-readable sanity check.',
      accent: SIG.teal,
    },
  ]

  const containerRef = useSlideTimeline((tl, el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-card', { autoAlpha: 0, y: 24, stagger: 0.09, duration: 0.5 }, 0.45)
      .from('.gs-card-num', { autoAlpha: 0, scale: 0.4, stagger: 0.09, duration: 0.35 }, 0.6)

    /* Stroke-draw on each card's small glyph, matching SlidePipeline's arrow pattern. */
    const glyphs = el.querySelectorAll<SVGPathElement>('.gs-glyph path')
    glyphs.forEach((p, i) => {
      const len = p.getTotalLength()
      tl.fromTo(p,
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 0.7, ease: 'power2.inOut' },
        0.8 + i * 0.09)
    })

    tl.from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="14" label="From Detection to Explanation" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Explainability
      </h2>
      <p className="gs-caption text-base text-ink-100 leading-relaxed mb-5 max-w-3xl">
        Detection hands us a month and a product. Explanation hands analysts a story they can act on.
      </p>

      <div className="grid grid-cols-2 grid-rows-2 gap-4 flex-1 min-h-0">
        {CARDS.map((c) => (
          <div key={c.num} className="gs-card relative bg-paper-100 border border-paper-300 rounded-xl p-5 flex gap-4 min-h-0 overflow-hidden">
            <span className="gs-card-num font-mono text-xs uppercase tracking-[0.18em] flex-shrink-0 w-7 pt-1" style={{ color: c.accent }}>{c.num}</span>
            <div className="flex-1 min-w-0 flex flex-col">
              <div className="flex items-baseline justify-between gap-2 mb-1.5">
                <h3 className="font-display text-2xl text-ink-300 leading-tight">{c.title}</h3>
                <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-50">{c.caption}</span>
              </div>
              <p className="text-sm text-ink-100 leading-relaxed">{c.body}</p>
              <svg className="gs-glyph mt-auto pt-3 self-end" width="82" height="28" viewBox="0 0 82 28" fill="none">
                {c.num === '01' && (
                  <>
                    <path d="M4 24 L22 24" stroke={c.accent} strokeWidth="2" strokeLinecap="round" />
                    <path d="M4 17 L34 17" stroke={c.accent} strokeWidth="2" strokeLinecap="round" />
                    <path d="M4 10 L56 10" stroke={c.accent} strokeWidth="2" strokeLinecap="round" />
                    <path d="M52 24 L78 24" stroke={c.accent} strokeWidth="2" strokeLinecap="round" opacity="0.35" />
                  </>
                )}
                {c.num === '02' && (
                  <>
                    <path d="M4 14 Q20 4 36 14 T68 14" stroke={c.accent} strokeWidth="1.6" strokeLinecap="round" />
                    <path d="M4 22 Q20 12 36 22 T68 22" stroke={c.accent} strokeWidth="1.6" strokeLinecap="round" opacity="0.55" />
                    <path d="M72 14 L78 14" stroke={c.accent} strokeWidth="1.6" strokeLinecap="round" />
                    <path d="M72 22 L78 22" stroke={c.accent} strokeWidth="1.6" strokeLinecap="round" opacity="0.55" />
                  </>
                )}
                {c.num === '03' && (
                  <>
                    <path d="M22 14 m -14 0 a 14 14 0 1 0 28 0 a 14 14 0 1 0 -28 0" stroke={c.accent} strokeWidth="1.6" fill="none" />
                    <path d="M40 14 m -14 0 a 14 14 0 1 0 28 0 a 14 14 0 1 0 -28 0" stroke={c.accent} strokeWidth="1.6" fill="none" opacity="0.55" />
                  </>
                )}
                {c.num === '04' && (
                  <>
                    <path d="M6 8 Q6 20 18 22" stroke={c.accent} strokeWidth="1.8" strokeLinecap="round" fill="none" />
                    <path d="M14 8 Q14 20 26 22" stroke={c.accent} strokeWidth="1.8" strokeLinecap="round" fill="none" />
                    <path d="M36 18 L74 18" stroke={c.accent} strokeWidth="1.6" strokeLinecap="round" opacity="0.5" />
                    <path d="M36 23 L64 23" stroke={c.accent} strokeWidth="1.6" strokeLinecap="round" opacity="0.35" />
                  </>
                )}
              </svg>
            </div>
          </div>
        ))}
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-4 justify-center">
        <span className="font-mono text-xs uppercase tracking-[0.2em] text-ink-50">Applied across</span>
        <span className="font-mono text-sm font-semibold text-ink-300">5 products</span>
        <span className="text-ink-50">&middot;</span>
        <span className="font-mono text-sm font-semibold text-ink-300">77 events</span>
        <span className="text-ink-50">&middot;</span>
        <span className="font-mono text-sm font-semibold text-ink-300">9 aspects</span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 7: Explainability (aspect delta + review evidence)
   ════════════════════════════════════════════════════════════════════ */
function Slide6() {
  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-delta-row', { autoAlpha: 0, x: -20, stagger: 0.1, duration: 0.4 }, 0.5)
      .from('.gs-dumbbell-line', { scaleX: 0, stagger: 0.1, duration: 0.4, ease: 'power2.out' }, 0.75)
      .from('.gs-after-dot', { autoAlpha: 0, scale: 0, transformOrigin: '50% 50%', stagger: 0.1, duration: 0.28 }, 0.95)
      .from('.gs-quote', { autoAlpha: 0, x: 20, stagger: 0.15, duration: 0.4 }, 0.8)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="15" label="Explainability" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Every alert speaks <span className="italic text-ink-100">customer language</span>
      </h2>
      <p className="gs-caption text-base text-ink-100 leading-relaxed mb-4">
        Aspect deltas and real review evidence ground every detection.
      </p>

      <div className="grid grid-cols-[1fr_1fr] gap-5 flex-1 min-h-0 items-center">
        {/* Aspect deltas — before/after dumbbell chart */}
        <div className="bg-paper-100 border border-paper-300 rounded-xl p-5 flex flex-col">
          <p className="font-mono text-xs uppercase tracking-wide text-ink-50 mb-3">Aspect Delta Ranking</p>

          {/* Column direction labels — structurally encode the before/after frame */}
          <div className="flex items-center gap-3 mb-2">
            <div className="w-3 flex-shrink-0" />
            <div className="w-[6.5rem] flex-shrink-0" />
            <div className="flex-1 flex justify-between">
              <span className="font-mono text-[9px] uppercase tracking-widest" style={{ color: SIG.red }}>After drift ←</span>
              <span className="font-mono text-[9px] uppercase tracking-widest" style={{ color: SIG.teal }}>→ Before drift</span>
            </div>
            <div className="w-14 flex-shrink-0" />
          </div>

          {/* Rows */}
          <div className="space-y-3.5">
            {DELTAS.map(({ a, d }) => {
              const BEFORE_VAL = 0.30
              const afterVal = BEFORE_VAL + d
              const scaleMin = -1.1, scaleMax = 0.6
              const toPos = (v: number) => ((v - scaleMin) / (scaleMax - scaleMin)) * 100
              const beforePct = toPos(BEFORE_VAL)
              const afterPct = toPos(afterVal)
              return (
                <div key={a} className="gs-delta-row flex items-center gap-3">
                  <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: AC[a] || INK.med }} />
                  <span className="text-sm text-ink-200 w-[6.5rem] capitalize">{a}</span>
                  <div className="flex-1 relative h-4 flex items-center">
                    {/* Track */}
                    <div className="absolute inset-0 flex items-center pointer-events-none">
                      <div className="w-full h-px" style={{ backgroundColor: PAP.stone }} />
                    </div>
                    {/* Connecting line — draws right→left on animate */}
                    <div className="gs-dumbbell-line absolute h-[2px] rounded-full" style={{
                      left: `${afterPct}%`,
                      width: `${beforePct - afterPct}%`,
                      backgroundColor: SIG.red,
                      opacity: 0.3,
                      transformOrigin: '100% 50%',
                    }} />
                    {/* BEFORE dot (teal) */}
                    <div className="absolute w-[11px] h-[11px] rounded-full" style={{
                      left: `${beforePct}%`,
                      transform: 'translateX(-50%)',
                      backgroundColor: SIG.teal,
                      boxShadow: `0 0 0 2px #F0ECE5, 0 0 0 3.5px ${SIG.teal}`,
                    }} />
                    {/* AFTER dot (red) */}
                    <div className="gs-after-dot absolute w-[11px] h-[11px] rounded-full" style={{
                      left: `${afterPct}%`,
                      transform: 'translateX(-50%)',
                      backgroundColor: SIG.red,
                      boxShadow: `0 0 0 2px #F0ECE5, 0 0 0 3.5px ${SIG.red}`,
                    }} />
                  </div>
                  <span className="font-mono text-sm w-14 text-right" style={{ color: SIG.red }}>
                    {d.toFixed(2)}
                  </span>
                </div>
              )
            })}
          </div>

          {/* Axis ticks */}
          <div className="gs-stat flex items-center gap-3 mt-3">
            <div className="w-3 flex-shrink-0" />
            <div className="w-[6.5rem] flex-shrink-0" />
            <div className="flex-1 flex justify-between font-mono text-[9px]" style={{ color: INK.dim }}>
              <span>–1.0</span><span>–0.5</span><span>0</span><span>+0.5</span>
            </div>
            <div className="w-14 flex-shrink-0" />
          </div>
        </div>

        {/* Review evidence */}
        <div className="flex flex-col gap-3 justify-center min-h-0 overflow-hidden">
          <div className="gs-quote px-5 py-4 rounded-xl bg-paper-100 border border-paper-300">
            <div className="flex items-center gap-2.5 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs text-white font-medium" style={{ backgroundColor: AC.connectivity }}>connectivity</span>
              <span className="text-xs text-ink-50">★★★ &middot; 2016-03</span>
            </div>
            <p className="text-sm text-ink-200 leading-relaxed italic">
              &ldquo;Bluetooth keeps disconnecting from my car. Have to reboot every time...&rdquo;
            </p>
          </div>
          <div className="gs-quote px-5 py-4 rounded-xl bg-paper-100 border border-paper-300">
            <div className="flex items-center gap-2.5 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs text-white font-medium" style={{ backgroundColor: AC.battery }}>battery</span>
              <span className="text-xs text-ink-50">★★ &middot; 2016-03</span>
            </div>
            <p className="text-sm text-ink-200 leading-relaxed italic">
              &ldquo;Battery life is terrible now. Heats up constantly. Falls WAY short of expectations.&rdquo;
            </p>
          </div>
          <div className="gs-quote px-5 py-4 rounded-xl bg-paper-100 border border-paper-300">
            <div className="flex items-center gap-2.5 mb-2">
              <span className="px-2.5 py-0.5 rounded-full text-xs text-white font-medium" style={{ backgroundColor: AC.audio }}>audio</span>
              <span className="text-xs text-ink-50">★★ &middot; 2016-04</span>
            </div>
            <p className="text-sm text-ink-200 leading-relaxed italic">
              &ldquo;Speaker crackles at any volume. Previous phone was much better.&rdquo;
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Integrated Gradients (token-level attribution)
   Real output from src/explain/captum_ig.py · 50 IG steps ·
   LayerIntegratedGradients on PyABSA APC word embeddings.
   Source: results/explainability/ig_demo_tokens.json
   ════════════════════════════════════════════════════════════════════ */
function SlideIGAttribution() {
  /* Scores are attribution-to-predicted-class, max-abs normalized per review
     by the Python exporter. Trailing LCF aspect-marker token stripped. */
  type IGRow = {
    aspect: keyof typeof AC
    label: 'negative' | 'positive'
    confidence: number
    tokens: string[]
    scores: number[]
  }
  const IG: IGRow[] = [
    {
      aspect: 'battery', label: 'negative', confidence: 0.97,
      tokens: ['The', 'battery', 'dies', 'after', 'only', '2', 'hours', '.', 'I', 'need', 'to', 'charge', 'it', 'constantly', '.'],
      scores: [0.241, 0.424, 1.0, 0.501, 0.594, 0.275, 0.239, 0.308, 0.104, 0.167, 0.084, 0.109, 0.086, 0.134, 0.202],
    },
    {
      aspect: 'camera', label: 'positive', confidence: 1.00,
      tokens: ['The', 'camera', 'is', 'still', 'excellent', ';', 'photos', 'look', 'great', 'in', 'daylight', '.'],
      scores: [-0.339, 0.281, -0.169, 0.819, 0.926, 0.808, 0.094, 0.688, 1.0, 0.094, -0.048, 0.258],
    },
    {
      aspect: 'software', label: 'negative', confidence: 0.99,
      tokens: ['Software', 'update', 'made', 'everything', 'slower', 'and', 'the', 'phone', 'restarts', 'randomly', '.'],
      scores: [0.292, 0.393, 0.414, 0.613, 1.0, 0.251, 0.189, 0.071, 0.176, 0.053, 0.141],
    },
  ]

  const containerRef = useSlideTimeline((tl) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-ig-row', { autoAlpha: 0, y: 16, stagger: 0.18, duration: 0.45 }, 0.45)
      .from('.gs-ig-tok', { autoAlpha: 0, stagger: 0.012, duration: 0.25 }, 0.75)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="07" label="Token Attribution" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Which words <span className="italic text-ink-100">drove the call?</span>
      </h2>
      <p className="gs-caption text-base text-ink-100 leading-relaxed mb-5">
        Layer Integrated Gradients<span className="font-mono text-[11px] font-semibold text-ink-50 align-super ml-0.5">[7,8]</span> on the PyABSA APC encoder. Per-token evidence, aspect-conditioned, 50 steps.
      </p>

      <div className="flex flex-col gap-3 flex-1 min-h-0 overflow-hidden">
        {IG.map(r => {
          const bandColor = r.label === 'positive' ? SIG.teal : SIG.red
          return (
            <div key={r.aspect} className="gs-ig-row bg-paper-100 border border-paper-300 rounded-xl px-5 py-4 flex flex-col gap-2">
              <div className="flex items-center gap-3">
                <span className="px-2.5 py-0.5 rounded-full text-xs text-white font-medium capitalize" style={{ backgroundColor: AC[r.aspect] }}>
                  {r.aspect}
                </span>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] font-semibold" style={{ color: bandColor }}>
                  {r.label}
                </span>
                <span className="font-mono text-xs text-ink-50">p = {r.confidence.toFixed(2)}</span>
              </div>
              <p className="text-lg md:text-xl leading-relaxed text-ink-200">
                {r.tokens.map((t, i) => {
                  const s = r.scores[i]
                  const mag = Math.max(0, s)
                  const alpha = Math.round(mag * 0.55 * 255).toString(16).padStart(2, '0')
                  const punct = /^[.,;!?]$/.test(t)
                  const bg = mag > 0.05 ? `${bandColor}${alpha}` : 'transparent'
                  return (
                    <span key={i}>
                      {!punct && i > 0 ? ' ' : ''}
                      <span className="gs-ig-tok rounded px-0.5" style={{ backgroundColor: bg }}>{t}</span>
                    </span>
                  )
                })}
              </p>
            </div>
          )
        })}
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-3 justify-center">
        <span className="font-mono text-xs uppercase tracking-[0.2em] text-ink-50">Captum</span>
        <span className="text-ink-50">&middot;</span>
        <span className="font-mono text-xs text-ink-100">3 reviews</span>
        <span className="text-ink-50">&middot;</span>
        <span className="font-mono text-xs text-ink-100">all predictions agree with human label</span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Aspect frequency across all detected events
   Real counts from results/explainability/product_*.json (top-1 aspect
   per event, 77 events total across 5 case-study products).
   ════════════════════════════════════════════════════════════════════ */
function SlideAspectFrequency() {
  /* Top-1 aspect frequency across 77 events, computed from product JSONs. */
  const FREQ: Array<{ a: keyof typeof AC; n: number }> = [
    { a: 'build', n: 18 },
    { a: 'connectivity', n: 17 },
    { a: 'audio', n: 13 },
    { a: 'camera', n: 11 },
    { a: 'software', n: 7 },
    { a: 'charging', n: 4 },
    { a: 'display', n: 3 },
    { a: 'battery', n: 2 },
    { a: 'performance', n: 2 },
  ]
  const TOTAL = FREQ.reduce((s, r) => s + r.n, 0)
  const maxN = FREQ[0].n
  const topThreeShare = Math.round(((FREQ[0].n + FREQ[1].n + FREQ[2].n) / TOTAL) * 100)

  const containerRef = useSlideTimeline((tl, el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-freq-band', { autoAlpha: 0, scaleY: 0.6, duration: 0.45, transformOrigin: 'center' }, 0.45)
      .from('.gs-bar-row', { autoAlpha: 0, x: -14, stagger: 0.07, duration: 0.38 }, 0.55)
      .from('.gs-bar-fill', { scaleX: 0, stagger: 0.07, duration: 0.65, ease: 'power2.out', transformOrigin: 'left center' }, 0.6)
      .from('.gs-bar-count', { autoAlpha: 0, x: -6, stagger: 0.07, duration: 0.3 }, 0.8)

    /* Count-up on the hero "62%" number. */
    const heroEl = el.querySelector<HTMLSpanElement>('.gs-hero-num')
    if (heroEl) {
      const obj = { v: 0 }
      tl.to(obj, {
        v: topThreeShare,
        duration: 1.0,
        ease: 'power2.out',
        onUpdate: () => { heroEl.textContent = Math.round(obj.v).toString() },
      }, 0.9)
    }

    tl.from('.gs-hero-frame', { autoAlpha: 0, y: 12, duration: 0.45 }, 0.7)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="16" label="Aspect Frequency" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Hardware aspects <span className="italic text-ink-100">dominate</span> drift
      </h2>
      <p className="gs-caption text-base text-ink-100 leading-relaxed mb-5">
        Across 77 detected events, build, connectivity and audio are the top-ranked driver in nearly two thirds of cases.
      </p>

      <div className="grid grid-cols-[1.4fr_1fr] gap-6 flex-1 min-h-0 items-center">

        {/* Bar chart: top-1 aspect per event */}
        <div className="relative bg-paper-100 border border-paper-300 rounded-xl p-5 flex flex-col justify-center min-h-0">
          <div className="flex items-baseline justify-between mb-4">
            <p className="font-mono text-xs uppercase tracking-[0.18em] text-ink-50">Rank-1 driver · 77 events</p>
            <p className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-50">count</p>
          </div>

          {/* Highlight band behind top-3 rows: the narrative anchor */}
          <div
            className="gs-freq-band absolute left-3 right-3 rounded-lg pointer-events-none"
            style={{
              top: 'calc(3rem + 4px)',
              height: 'calc(3 * (1.15rem + 10px))',
              background: `${SIG.teal}10`,
              border: `1px dashed ${SIG.teal}30`,
            }}
          />

          <div className="relative space-y-2.5">
            {FREQ.map(({ a, n }) => {
              const pct = (n / TOTAL) * 100
              return (
                <div key={a} className="gs-bar-row flex items-center gap-3">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: AC[a] }} />
                  <span className="text-sm text-ink-200 w-24 capitalize">{a}</span>
                  <div className="flex-1 h-3 bg-paper-200 rounded-full overflow-hidden">
                    <div
                      className="gs-bar-fill h-full rounded-full"
                      style={{
                        width: `${(n / maxN) * 100}%`,
                        backgroundColor: AC[a],
                        opacity: n >= FREQ[2].n ? 1 : 0.55,
                      }}
                    />
                  </div>
                  <span className="gs-bar-count font-mono text-sm text-ink-300 w-14 text-right tabular-nums">
                    {n}<span className="text-ink-50 ml-1 text-[11px]">·{pct.toFixed(0)}%</span>
                  </span>
                </div>
              )
            })}
          </div>
        </div>

        {/* Hero call-out */}
        <div className="gs-hero-frame flex flex-col gap-4">
          <div className="bg-paper-100 border border-paper-300 rounded-xl p-6">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-2">Top-3 concentration</p>
            <div className="flex items-baseline gap-2">
              <span className="gs-hero-num font-display text-7xl leading-none" style={{ color: SIG.teal }}>{topThreeShare}</span>
              <span className="font-display text-4xl leading-none text-ink-200">%</span>
            </div>
            <p className="text-sm text-ink-100 leading-relaxed mt-3">
              of all top-ranked drift drivers are <span className="font-medium text-ink-200">build</span>,{' '}
              <span className="font-medium text-ink-200">connectivity</span> or{' '}
              <span className="font-medium text-ink-200">audio</span>.
            </p>
          </div>

          <div className="bg-paper-100 border border-paper-300 rounded-xl p-5">
            <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-2">Why this matters</p>
            <p className="text-sm text-ink-100 leading-relaxed">
              Drift is not random noise; radio and physical-build issues concentrate where warranties and returns hurt most.
              A monitoring spend targeted at these aspects recovers the majority of risk.
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 8: Business Value
   ════════════════════════════════════════════════════════════════════ */
function SlideBusinessValue() {
  const W = 560, H = 180
  const X0 = 50, X1 = 530, Y0 = 15, Y1 = 155
  const vMin = 0, vMax = 0.85

  /* Shared pre-drift sentiment (months 0–10) */
  const SHARED = [.68, .70, .65, .72, .67, .69, .71, .66, .68, .64, .55]
  /* Post-drift without monitoring: continued decline */
  const DECLINE = [.55, .48, .41, .35, .30, .26, .23, .21]
  /* Post-drift with SentinelDrift: alert, fix, recovery */
  const RECOVER = [.55, .50, .46, .48, .54, .59, .62, .64]

  const xDrift = X0 + (10 / 17) * (X1 - X0)
  const sharedP = dataPath(SHARED, X0, xDrift, Y0, Y1, vMin, vMax)
  const declineP = dataPath(DECLINE, xDrift, X1, Y0, Y1, vMin, vMax)
  const recoverP = dataPath(RECOVER, xDrift, X1, Y0, Y1, vMin, vMax)

  /* Fill polygon between the two futures */
  const fillPoly = (() => {
    const pts: string[] = []
    for (let i = 0; i < RECOVER.length; i++) {
      const x = xDrift + (i / (RECOVER.length - 1)) * (X1 - xDrift)
      const y = Y0 + (1 - (RECOVER[i] - vMin) / (vMax - vMin)) * (Y1 - Y0)
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`)
    }
    for (let i = DECLINE.length - 1; i >= 0; i--) {
      const x = xDrift + (i / (DECLINE.length - 1)) * (X1 - xDrift)
      const y = Y0 + (1 - (DECLINE[i] - vMin) / (vMax - vMin)) * (Y1 - Y0)
      pts.push(`${x.toFixed(1)},${y.toFixed(1)}`)
    }
    return pts.join(' ')
  })()

  /* Key coordinates */
  const yAt = (v: number) => Y0 + (1 - (v - vMin) / (vMax - vMin)) * (Y1 - Y0)
  const driftY = yAt(0.55)
  const fixIdx = 2                 // month 12: fix shipped
  const fixX = xDrift + (fixIdx / (RECOVER.length - 1)) * (X1 - xDrift)
  const fixY = yAt(RECOVER[fixIdx])
  const decEndY = yAt(DECLINE[DECLINE.length - 1])
  const recEndY = yAt(RECOVER[RECOVER.length - 1])

  const containerRef = useSlideTimeline((tl, el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-chart', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.45)

    /* Draw shared history */
    const shared = el.querySelector('.gs-shared-line') as SVGPathElement | null
    if (shared) {
      const len = shared.getTotalLength()
      tl.fromTo('.gs-shared-line',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.2, ease: 'power2.inOut' }, 0.6)
    }

    /* Detection pulse */
    tl.from('.gs-detect', { autoAlpha: 0, scale: 0, duration: 0.4, svgOrigin: `${xDrift} ${driftY}` }, 1.6)
      .to('.gs-detect-ring', { scale: 2, autoAlpha: 0, duration: 0.8, ease: 'power1.out', svgOrigin: `${xDrift} ${driftY}` }, 1.8)

    /* Both futures appear after detection pulse */
    tl.from('.gs-decline-line', { autoAlpha: 0, duration: 0.8 }, 2.0)
    const rec = el.querySelector('.gs-recover-line') as SVGPathElement | null
    if (rec) {
      const len = rec.getTotalLength()
      tl.fromTo('.gs-recover-line',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.0, ease: 'power2.inOut' }, 2.0)
    }

    tl.from('.gs-fix-marker', { autoAlpha: 0, scale: 0, duration: 0.3 }, 2.6)
      .from('.gs-fill-area', { autoAlpha: 0, duration: 0.6 }, 2.8)
      .from('.gs-end-label', { autoAlpha: 0, x: -8, stagger: 0.1, duration: 0.3 }, 3.2)
      .from('.gs-stats-overlay', { autoAlpha: 0, y: -8, duration: 0.4 }, '>-0.3')
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="18" label="Business Value" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Early Intervention
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-5">
        The shaded gap is the cost of not monitoring.
      </p>

      <div className="gs-chart relative flex-1 bg-paper-100 border border-paper-300 rounded-xl p-4 overflow-hidden">
        <div className="gs-stats-overlay absolute top-3 right-3 text-right text-xs text-ink-50 leading-relaxed">
          <div><span className="font-mono font-semibold text-ink-300">335</span> products &middot; <span className="font-mono font-semibold text-ink-300">9</span> aspects &middot; real-time</div>
        </div>
        <div className="flex gap-4 mb-2 text-xs text-ink-50">
          <span className="flex items-center gap-2"><span className="w-4 h-0.5 inline-block rounded" style={{ backgroundColor: INK.deep }} /> Shared history</span>
          <span className="flex items-center gap-2"><span className="w-4 border-t border-dashed inline-block" style={{ borderColor: PAP.stone }} /> Without monitoring</span>
          <span className="flex items-center gap-2"><span className="w-4 h-0.5 inline-block rounded" style={{ backgroundColor: SIG.teal }} /> With SentinelDrift</span>
        </div>
        <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="overflow-visible">
          {/* Grid */}
          {[0, 0.2, 0.4, 0.6, 0.8].map(v => (
            <line key={v} x1={X0} x2={X1} y1={yAt(v)} y2={yAt(v)} stroke={PAP.bone} strokeDasharray="3 3" />
          ))}
          {[0, 0.4, 0.8].map(v => (
            <text key={v} x={X0 - 8} y={yAt(v) + 3} textAnchor="end" fontSize="9" fill={INK.dim} fontFamily="JetBrains Mono">{v.toFixed(1)}</text>
          ))}
          {[0, 4, 8, 10, 14, 17].map(i => (
            <text key={i} x={X0 + (i / 17) * (X1 - X0)} y={Y1 + 14} textAnchor="middle" fontSize="8" fill={i === 10 ? SIG.red : INK.dim} fontFamily="JetBrains Mono">{`M${i}`}</text>
          ))}

          {/* Vertical split at drift point */}
          <line x1={xDrift} x2={xDrift} y1={Y0} y2={Y1} stroke={SIG.red} strokeWidth={0.75} strokeDasharray="3 3" opacity={0.25} />

          {/* Fill polygon: the protected value */}
          <polygon className="gs-fill-area" points={fillPoly} fill={SIG.teal} opacity={0.08} />

          {/* Shared history */}
          <path className="gs-shared-line" d={sharedP} fill="none" stroke={INK.deep} strokeWidth={2} strokeLinecap="round" />

          {/* Without monitoring: declining */}
          <path className="gs-decline-line" d={declineP} fill="none" stroke={PAP.stone} strokeWidth={1.5} strokeDasharray="6 4" strokeLinecap="round" />

          {/* With SentinelDrift: recovery */}
          <path className="gs-recover-line" d={recoverP} fill="none" stroke={SIG.teal} strokeWidth={2.5} strokeLinecap="round" />

          {/* Detection marker */}
          <g className="gs-detect">
            <circle className="gs-detect-ring" cx={xDrift} cy={driftY} r={8} fill="none" stroke={SIG.red} strokeWidth={2} />
            <circle cx={xDrift} cy={driftY} r={4} fill={SIG.red} />
            <text x={xDrift} y={Y0 - 4} textAnchor="middle" fontSize="10" fontWeight="600" fill={SIG.red} fontFamily="JetBrains Mono">DETECTED</text>
          </g>

          {/* Fix-shipped marker */}
          <g className="gs-fix-marker">
            <circle cx={fixX} cy={fixY} r={3.5} fill={SIG.teal} />
            <text x={fixX} y={fixY - 10} textAnchor="middle" fontSize="8" fontWeight="500" fill={SIG.teal} fontFamily="JetBrains Mono">FIX SHIPPED</text>
          </g>

          {/* End-of-path outcomes */}
          <g className="gs-end-label">
            <text x={X1 + 8} y={recEndY + 3} fontSize="9" fill={SIG.teal} fontFamily="JetBrains Mono" fontWeight="600">stable</text>
          </g>
          <g className="gs-end-label">
            <text x={X1 + 8} y={decEndY + 3} fontSize="9" fill={SIG.red} fontFamily="JetBrains Mono" fontWeight="500" opacity={0.55}>crisis</text>
          </g>
        </svg>
      </div>

    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE 9: CTA
   ════════════════════════════════════════════════════════════════════ */
function Slide7({ onEnter }: { onEnter: () => void }) {
  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, y: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 30, duration: 0.6 }, 0.2)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.4)
      .from('.gs-cta', { autoAlpha: 0, y: 16, scale: 0.95, duration: 0.5 }, 0.6)
      .from('.gs-hint', { autoAlpha: 0, duration: 0.3 }, 0.9)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col items-center justify-center text-center py-8">
      <p className="gs-label font-mono text-xs uppercase tracking-[0.2em] text-ink-50 mb-8">28 &middot; Live Demo</p>
      <h2 className="gs-heading font-display text-5xl md:text-6xl leading-tight mb-10">
        Now see it <span className="italic text-ink-100">in action</span>
      </h2>
      <motion.button onClick={onEnter}
        className="gs-cta px-10 py-4 rounded-xl bg-ink-300 text-paper-50 font-medium text-lg hover:bg-ink-200 transition-colors"
        whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
      >
        Enter IntelliSys Workspace
      </motion.button>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Title
   ════════════════════════════════════════════════════════════════════ */
function SlideTitle() {
  const TEAM = ['Ghulam Ahmed', 'Dilara Bayram', 'Ubay Alshamali', 'Adam Okeahalam', 'Alex Young', 'Nicholas Saremi']

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-logo', { autoAlpha: 0, y: -12, duration: 0.4 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 30, duration: 0.6 }, 0.25)
      .from('.gs-sub', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.5)
      .from('.gs-divider', { scaleX: 0, duration: 0.5 }, 0.65)
      .from('.gs-meta', { autoAlpha: 0, y: 12, duration: 0.35 }, 0.8)
      .from('.gs-member', { autoAlpha: 0, y: 10, stagger: 0.08, duration: 0.3 }, 1.0)
      .from('.gs-date', { autoAlpha: 0, duration: 0.3 }, 1.5)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-row items-center gap-12">
      {/* Left: text content */}
      <div className="flex flex-col items-start text-left" style={{ width: 340, flexShrink: 0 }}>
        <p className="gs-logo font-mono text-xs uppercase tracking-[0.25em] text-signal-teal mb-6">IntelliSys Ltd.</p>
        <h1 className="gs-heading font-display text-5xl md:text-6xl leading-tight mb-3">
          Sentinel<span className="italic text-ink-100">Drift</span>
        </h1>
        <p className="gs-sub text-xl md:text-2xl text-ink-100 leading-relaxed mb-8">
          Temporal Sentiment Drift Detection for Product Quality
        </p>
        <div className="gs-divider h-px w-48 bg-paper-400 mb-8" style={{ transformOrigin: 'left' }} />
        <p className="gs-meta font-mono text-xs text-ink-50 uppercase tracking-[0.15em] mb-6">
          WM9B7 AI &amp; Deep Learning
        </p>
        <div className="flex flex-wrap gap-x-5 gap-y-2 max-w-sm">
          {TEAM.map(name => (
            <span key={name} className="gs-member text-sm text-ink-200">{name}</span>
          ))}
        </div>
        <p className="gs-date font-mono text-xs text-ink-50 mt-8">April 2026</p>
      </div>
      {/* Right: group photo */}
      <div className="gs-date flex-1 flex items-center justify-center">
        <img
          src="/images/group.jpg"
          alt="Team photo"
          className="rounded-2xl object-cover shadow-lg"
          style={{ height: 360, width: '100%', maxWidth: 560 }}
        />
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Overall Pipeline (raw reviews → early warnings)
   Inserted between Problem (03) and Dataset (05) to frame the system
   before any EDA detail lands. Horizontal 5-stage flow, SVG, GSAP
   stroke-draw arrows + glyph reveals.
   ════════════════════════════════════════════════════════════════════ */
function SlidePipeline() {
  /* Layout geometry. Wider 1200×235 viewBox (≈5.1:1, same ratio as original
     960×188) gives arrow gaps enough room for "time series"/"changepoint"
     labels while keeping box proportions unchanged. CY=136 centres content. */
  const BOX_W = 152, BOX_H = 80, CY = 136
  type Phase = {
    cx: number; num: string; name: string; sub: string;
    carries: string; metric: string; color: string;
    detectors?: Array<{ name: string; color: string }>
  }
  const PHASES: Phase[] = [
    { cx: 120, num: '00', name: 'Ingest', sub: 'EDA', carries: 'raw text', metric: '312,100 reviews', color: SIG.red },
    { cx: 360, num: '01', name: 'Extract', sub: 'ABSA', carries: 'aspects', metric: '(aspect, sentiment)', color: AC.camera },
    { cx: 600, num: '02', name: 'Aggregate', sub: 'Features', carries: 'time series', metric: '335 × 24 monthly bins', color: DET.cusum },
    {
      cx: 840, num: '03', name: 'Detect', sub: 'Changepoint', carries: 'changepoint', metric: '3 methods compared', color: DET.lstm,
      detectors: [
        { name: 'CUSUM', color: DET.cusum },
        { name: 'LSTM-AE', color: DET.lstm },
        { name: 'PELT', color: INK.med },
      ]
    },
    { cx: 1080, num: '04', name: 'Explain', sub: 'Attribution', carries: '', metric: 'Δ · evidence · trajectory', color: SIG.teal },
  ]

  /* Arrow endpoints: box-right-edge → next-box-left-edge, with a 6-unit gap for the head. */
  const ARROWS = PHASES.slice(0, -1).map((p, i) => {
    const next = PHASES[i + 1]
    return { x1: p.cx + BOX_W / 2 + 4, x2: next.cx - BOX_W / 2 - 6, y: CY, color: next.color, carries: p.carries }
  })

  /* Glyph renderers: tiny 88×26 motifs showing the dataflow transformation above each arrow.
     GY=44 → glyph spans 44–70, leaving 26 units of clearance above the box top at y=96. */
  type Glyph = (cx: number, color: string) => React.ReactNode
  const GY = 44
  const GLYPHS: Glyph[] = [
    /* After Ingest → review text lines */
    (cx, c) => (
      <g transform={`translate(${cx - 44}, ${GY})`} stroke={c} strokeLinecap="round">
        <line x1={10} y1={6} x2={78} y2={6} strokeWidth={1.6} opacity={0.5} />
        <line x1={10} y1={14} x2={66} y2={14} strokeWidth={1.6} opacity={0.7} />
        <line x1={10} y1={22} x2={52} y2={22} strokeWidth={1.6} />
      </g>
    ),
    /* After Extract → three aspect dots (battery, camera, software) */
    (cx) => (
      <g transform={`translate(${cx - 44}, ${GY})`}>
        <circle cx={22} cy={14} r={5} fill={AC.battery} />
        <circle cx={44} cy={14} r={5} fill={AC.camera} />
        <circle cx={66} cy={14} r={5} fill={AC.software} />
      </g>
    ),
    /* After Aggregate → monthly zigzag */
    (cx, c) => (
      <g transform={`translate(${cx - 44}, ${GY})`}>
        <polyline
          points="8,18 20,9 32,16 44,7 56,20 68,11 80,15"
          fill="none" stroke={c} strokeWidth={1.7}
          strokeLinecap="round" strokeLinejoin="round"
        />
      </g>
    ),
    /* After Detect → zigzag with a red changepoint tick */
    (cx, c) => (
      <g transform={`translate(${cx - 44}, ${GY})`}>
        <polyline
          points="8,18 20,9 32,16 44,7 56,20 68,11 80,15"
          fill="none" stroke={c} strokeWidth={1.7}
          strokeLinecap="round" strokeLinejoin="round" opacity={0.45}
        />
        <line x1={50} y1={0} x2={50} y2={26} stroke={SIG.red} strokeWidth={1.8} strokeLinecap="round" />
        <circle cx={50} cy={0} r={2.6} fill={SIG.red} />
      </g>
    ),
  ]

  const containerRef = useSlideTimeline((tl, el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-phase', {
        autoAlpha: 0, scale: 0.88, transformOrigin: 'center center',
        stagger: 0.1, duration: 0.45
      }, 0.5)

    /* Arrow stroke-draw. Per-path length from getTotalLength, matching Slide2's pattern. */
    const arrows = el.querySelectorAll<SVGPathElement>('.gs-arrow')
    arrows.forEach((p, i) => {
      const len = p.getTotalLength()
      tl.fromTo(p,
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 0.45, ease: 'power2.inOut' },
        0.62 + i * 0.1,
      )
    })

    tl.from('.gs-arrow-head', {
      autoAlpha: 0, scale: 0.6, transformOrigin: 'center center',
      stagger: 0.1, duration: 0.25
    }, 0.8)
      .from('.gs-glyph', { autoAlpha: 0, y: 6, stagger: 0.1, duration: 0.3 }, 0.9)
      .from('.gs-carries', { autoAlpha: 0, y: -4, stagger: 0.1, duration: 0.3 }, 1.0)
      .from('.gs-metric', { autoAlpha: 0, y: 4, stagger: 0.08, duration: 0.3 }, 1.1)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="04" label="Pipeline" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        From raw reviews to <span className="italic text-ink-100">early warnings</span>
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-5">
        Five stages turn 312,100 Amazon reviews into per-aspect drift signals.
      </p>

      <div className="flex-1 flex items-center min-h-0">
        <div className="w-full bg-paper-100 border border-paper-300 rounded-xl px-8 py-7 overflow-hidden">
          <svg viewBox="0 0 1200 235" width="100%" className="overflow-visible block">
            {/* Arrows: drawn under the boxes so heads meet the border cleanly */}
            {ARROWS.map((a, i) => (
              <g key={`arrow-${i}`}>
                <path
                  className="gs-arrow"
                  d={`M ${a.x1} ${a.y} L ${a.x2} ${a.y}`}
                  stroke={a.color} strokeWidth={1.8} fill="none" strokeLinecap="round"
                  opacity={0.8}
                />
                {/* Arrowhead: tapered triangle, sized to match the bolder stroke */}
                <polygon
                  className="gs-arrow-head"
                  points={`${a.x2},${a.y - 5} ${a.x2 + 8},${a.y} ${a.x2},${a.y + 5}`}
                  fill={a.color}
                />
              </g>
            ))}

            {/* Glyphs: tiny dataflow transformation motifs centred above each arrow */}
            {GLYPHS.map((g, i) => {
              const cx = (ARROWS[i].x1 + ARROWS[i].x2) / 2
              return <g key={`glyph-${i}`} className="gs-glyph">{g(cx, ARROWS[i].color)}</g>
            })}

            {/* Carries label: what the arrow transports (italic serif, ink-dim) */}
            {ARROWS.map((a, i) => {
              const cx = (a.x1 + a.x2) / 2
              return (
                <text key={`carries-${i}`}
                  className="gs-carries"
                  x={cx} y={a.y + 16}
                  textAnchor="middle"
                  fontFamily="Instrument Serif, serif"
                  fontStyle="italic" fontSize="11.5"
                  fill={INK.dim}>
                  {a.carries}
                </text>
              )
            })}

            {/* Phase boxes */}
            {PHASES.map(p => (
              <g key={p.num} className="gs-phase" transform={`translate(${p.cx}, ${CY})`}>
                {/* Subtle colour-washed fill gives each phase its own DNA without shouting */}
                <rect
                  x={-BOX_W / 2} y={-BOX_H / 2} width={BOX_W} height={BOX_H} rx={10}
                  fill={`${p.color}10`} stroke={p.color} strokeWidth={1.5}
                />
                {/* Phase number: mono, coloured, heavier weight */}
                <text x={-BOX_W / 2 + 12} y={-BOX_H / 2 + 18}
                  fontFamily="JetBrains Mono" fontSize="10" fontWeight="700"
                  fill={p.color} letterSpacing="0.08em">
                  {p.num}
                </text>
                {/* Sub-label: uppercase mono on the right */}
                <text x={BOX_W / 2 - 12} y={-BOX_H / 2 + 18}
                  textAnchor="end"
                  fontFamily="JetBrains Mono" fontSize="8"
                  fill={INK.dim} letterSpacing="0.18em">
                  {p.sub.toUpperCase()}
                </text>
                {/* Phase name: display serif, centred. For the Detect phase we
                  swap the single label for three stacked detector pills so the
                  comparison (CUSUM vs LSTM-AE vs PELT) reads at a glance. */}
                {p.detectors ? (
                  p.detectors.map((d, i) => {
                    const cy = -9 + i * 17
                    return (
                      <g key={d.name} className="gs-detector">
                        <rect
                          x={-52} y={cy - 7.5} width={104} height={15} rx={4}
                          fill={`${d.color}18`} stroke={d.color} strokeWidth={1}
                        />
                        <text x={0} y={cy + 3.2}
                          textAnchor="middle"
                          fontFamily="JetBrains Mono" fontSize="9.5" fontWeight="700"
                          fill={d.color} letterSpacing="0.10em">
                          {d.name}
                        </text>
                      </g>
                    )
                  })
                ) : (
                  <text x={0} y={10}
                    textAnchor="middle"
                    fontFamily="Instrument Serif, serif"
                    fontSize="24" fontWeight="400" fill={INK.black}>
                    {p.name}
                  </text>
                )}
              </g>
            ))}

            {/* Per-phase metric beneath each box */}
            {PHASES.map(p => (
              <text key={`m-${p.num}`}
                className="gs-metric"
                x={p.cx} y={CY + BOX_H / 2 + 20}
                textAnchor="middle"
                fontFamily="JetBrains Mono" fontSize="10"
                fill={INK.med}>
                {p.metric}
              </text>
            ))}
          </svg>
        </div>
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-auto pt-5 justify-center">
        <span className="font-mono text-2xl font-semibold text-signal-red">312,100</span>
        <span className="text-sm text-ink-50">reviews</span>
        <svg className="text-ink-30 mx-2" width="22" height="12" viewBox="0 0 28 16" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 8h24M19 1l6 7-6 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="font-mono text-2xl font-semibold text-ink-300">335</span>
        <span className="text-sm text-ink-50">monitored products</span>
        <svg className="text-ink-30 mx-2" width="22" height="12" viewBox="0 0 28 16" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M1 8h24M19 1l6 7-6 7" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="font-mono text-2xl font-semibold text-signal-teal">5</span>
        <span className="text-sm text-ink-50 italic">case-study drifts, explained</span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Dataset & EDA
   ════════════════════════════════════════════════════════════════════ */
function SlideDataset() {
  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-funnel-row', { autoAlpha: 0, x: -20, stagger: 0.12, duration: 0.35 }, 0.4)
      .from('.gs-funnel-bar', { scaleX: 0, stagger: 0.12, duration: 0.55, ease: 'power2.out' }, 0.6)
      .from('.gs-eda-header', { autoAlpha: 0, y: 8, duration: 0.3 }, 0.45)
      .from('.gs-eda-phase-header', { autoAlpha: 0, x: -10, stagger: 0.1, duration: 0.3 }, 0.55)
      .from('.gs-eda-stage', { autoAlpha: 0, x: -8, stagger: 0.08, duration: 0.3 }, 0.7)
      .from('.gs-source', { autoAlpha: 0, y: 10, duration: 0.3 }, 1.4)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="05" label="Dataset" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-4">
        Amazon Reviews 2023 <span className="italic text-ink-100">(smartphones only)</span>
      </h2>

      <div className="flex-1 bg-paper-100 border border-paper-300 rounded-xl p-5 grid grid-cols-[0.8fr_1.2fr] gap-6 min-h-0 overflow-hidden">
        {/* Left: funnel */}
        <div className="flex flex-col justify-center min-w-0">
          <div className="space-y-4 w-full">
            {FUNNEL.map(({ label, value, bar }) => (
              <div key={label} className="gs-funnel-row">
                <div className="flex items-baseline justify-between mb-1.5">
                  <span className="text-sm text-ink-200 font-medium">{label}</span>
                  <span className="font-mono text-lg font-semibold text-ink-300">{value}</span>
                </div>
                <div className="h-3.5 bg-paper-200 rounded-full overflow-hidden">
                  <div className="gs-funnel-bar h-full rounded-full" style={{
                    width: `${Math.max(bar * 100, 2)}%`,
                    backgroundColor: bar > 0.1 ? SIG.teal : bar > 0.01 ? DET.cusum : SIG.red,
                    transformOrigin: 'left',
                  }} />
                </div>
              </div>
            ))}
          </div>
          <p className="gs-source text-[10px] text-ink-50 mt-5 leading-relaxed">
            Hou et al. (2024)<span className="font-mono font-semibold text-ink-200 align-super ml-0.5">[1]</span> &middot; MIT License
          </p>
        </div>

        {/* Right: EDA pipeline flow chart */}
        <div className="flex flex-col min-h-0 min-w-0 border-l border-paper-300 pl-5">
          <div className="gs-eda-header flex items-baseline justify-between mb-2">
            <span className="text-[11px] uppercase tracking-[0.18em] text-ink-200 font-semibold">EDA Pipeline</span>
          </div>
          <div className="flex-1 min-h-0 overflow-hidden flex flex-col gap-2">
            {EDA_PHASES.map((ph, phIdx) => (
              <div key={ph.phase} className="gs-eda-phase-block">
                <div className="gs-eda-phase-header flex items-center gap-2 mb-1">
                  <span className="font-mono text-[9px] font-semibold w-5" style={{ color: ph.color }}>
                    {String(phIdx + 1).padStart(2, '0')}
                  </span>
                  <span
                    className="text-[9px] uppercase tracking-[0.2em] font-semibold"
                    style={{ color: ph.color }}
                  >
                    {ph.phase}
                  </span>
                  <span className="flex-1 h-px" style={{ backgroundColor: ph.color, opacity: 0.35 }} />
                </div>
                <div className="gs-eda-stage px-2 py-1 space-y-0.5">
                  {ph.stages.map(stage => (
                    <div key={stage.group} className="flex items-center gap-2">
                      <span className="text-[10px] text-ink-300 font-semibold flex-shrink-0 w-[82px] truncate">
                        {stage.group}
                      </span>
                      <span className="text-[9px] text-ink-100 font-mono leading-tight truncate">
                        {stage.items.join(' · ')}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-auto pt-4 justify-center">
        <span className="font-mono text-3xl font-semibold text-signal-red">111</span>
        <span className="text-base text-ink-50">of 338 dense products show BH-FDR corrected significant drift <span className="text-ink-50/70">(monthly Mann&ndash;Whitney U, rolling &Delta; &ge; 0.5, |Cohen&#39;s d| &ge; 0.5)</span></span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: ABSA Evaluation
   ════════════════════════════════════════════════════════════════════ */
function SlideABSAEval() {
  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-example', { autoAlpha: 0, y: 12, stagger: 0.08, duration: 0.4 }, 0.5)
      .from('.gs-metric', { autoAlpha: 0, y: 12, stagger: 0.1, duration: 0.4 }, 0.95)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  const examples: { text: string; aspect: string; sentiment: 'positive' | 'negative' | 'neutral' }[] = [
    { text: '"Battery died after only 3 months."', aspect: 'battery', sentiment: 'negative' },
    { text: '"Touch screen does not work. Have to return."', aspect: 'display', sentiment: 'negative' },
    { text: '"Everything seemed good, battery life at 89%, nice Screen…"', aspect: 'display', sentiment: 'positive' },
    { text: '"The ring/silent switch broke within 5 months."', aspect: 'build', sentiment: 'negative' },
  ]

  const sentColor = (s: string) =>
    s === 'positive' ? 'text-signal-teal' : s === 'negative' ? 'text-signal-red' : 'text-ink-100'

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="08" label="ABSA Evaluation" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Evaluating against <span className="italic text-ink-100">gold labels</span>
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-5">
        PyABSA<span className="font-mono text-[11px] font-semibold text-ink-50 align-super ml-0.5">[2]</span> vs TF-IDF + Logistic Regression on 195 human-corrected gold triplets.
      </p>

      <div className="flex-1 bg-paper-100 border border-paper-300 rounded-xl p-5 grid grid-cols-[minmax(0,0.9fr)_minmax(0,1.4fr)] gap-6 min-h-0">
        <div className="flex flex-col gap-3 justify-center">
          <div className="gs-metric rounded-lg p-4"
            style={{ backgroundColor: `${SIG.teal}0F` }}>
            <div className="text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-1">PyABSA (DL)</div>
            <div className="font-mono text-3xl font-semibold text-signal-teal leading-none">0.819</div>
            <div className="text-[11px] text-ink-50 mt-1">CV Macro F1 &middot; [0.737, 0.888]</div>
          </div>
          <div className="gs-metric rounded-lg p-4 bg-paper-200">
            <div className="text-[10px] uppercase tracking-[0.2em] text-ink-50 mb-1">TF-IDF + LR (ML)</div>
            <div className="font-mono text-3xl font-semibold text-ink-200 leading-none">0.304</div>
            <div className="text-[11px] text-ink-50 mt-1">CV Macro F1 &middot; [0.236, 0.380]</div>
          </div>
        </div>

        <div className="flex flex-col min-w-0 justify-center">
          <div className="text-[11px] font-mono uppercase tracking-[0.2em] text-ink-50 mb-2">
            Gold triplets &nbsp;·&nbsp; review &rarr; (aspect, sentiment)
          </div>
          <div className="flex flex-col">
            {examples.map((e, i) => (
              <div key={i} className="gs-example flex items-center gap-3 py-2 border-b border-paper-300/70 last:border-0">
                <span className="flex-1 min-w-0 text-sm text-ink-200 italic leading-snug">{e.text}</span>
                <span className="font-mono text-xs text-ink-100 w-20 text-right shrink-0">{e.aspect}</span>
                <span className={`font-mono text-xs font-semibold w-20 text-right shrink-0 ${sentColor(e.sentiment)}`}>
                  {e.sentiment}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <StatLine className="justify-center !border-t-0">
        <span className="font-mono text-3xl font-semibold text-signal-teal">+51.5pp</span>
        <span className="text-base text-ink-50">DL advantage &middot; McNemar p &lt; 0.001 &middot; 5-fold StratifiedGroupKFold (n=195)</span>
      </StatLine>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Case Study (Huawei Nexus 6P, ASIN B019BX7F86)
   ════════════════════════════════════════════════════════════════════ */
function SlideCaseStudy() {
  const W = 560, H = 160
  const X0 = 50, X1 = 530, Y0 = 15, Y1 = 140

  const battPath = dataPath(NEXUS6P_BATTERY, X0, X1, Y0, Y1, -0.05, 0.95)
  const buildPath = dataPath(NEXUS6P_BUILD, X0, X1, Y0, Y1, -0.05, 0.95)

  /* Drift markers: LSTM fires Nov 2016 (month 10, first battery decay), CUSUM fires Mar 2017 (month 14, sustained decline) */
  const lstmX = X0 + (10 / 23) * (X1 - X0)
  const cusumX = X0 + (14 / 23) * (X1 - X0)

  /* Key dates for X-axis labels (24 monthly points: Jan 2016 – Dec 2017) */
  const X_LABELS: [number, string][] = [[0, '2016-01'], [6, '2016-07'], [12, '2017-01'], [18, '2017-07'], [23, '2017-12']]

  /* Review pointers: each pin is anchored to a data line so its Y comes
     directly from the series at that index. Performance is routed to the
     battery line (the device becoming a "paperweight" pulls battery sentiment
     to rock bottom) so it sits clearly below the build pin two months later. */
  const yAt = (v: number) => Y0 + (1 - (v + 0.05) / 1.0) * (Y1 - Y0)
  const ASPECT_SERIES: Record<'battery' | 'build', number[]> = {
    battery: NEXUS6P_BATTERY,
    build: NEXUS6P_BUILD,
  }
  const REVIEW_MARKERS: { idx: number; color: string; series: 'battery' | 'build' }[] = [
    { idx: 16, color: AC.battery, series: 'battery' }, /* 2017-05 battery */
    { idx: 20, color: AC.performance, series: 'battery' }, /* 2017-09 performance (paperweight → battery floor) */
    { idx: 22, color: AC.build, series: 'build' }, /* 2017-11 build */
  ]

  const containerRef = useSlideTimeline((tl, el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-chart', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.45)

    const batt = el.querySelector('.gs-batt-line') as SVGPathElement | null
    if (batt) {
      const len = batt.getTotalLength()
      tl.fromTo('.gs-batt-line',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.4, ease: 'power2.inOut' }, 0.6)
    }
    const build = el.querySelector('.gs-build-line') as SVGPathElement | null
    if (build) {
      const len = build.getTotalLength()
      tl.fromTo('.gs-build-line',
        { strokeDasharray: len, strokeDashoffset: len },
        { strokeDashoffset: 0, duration: 1.4, ease: 'power2.inOut' }, 0.7)
    }

    tl.from('.gs-drift-marker', { autoAlpha: 0, scaleY: 0, stagger: 0.15, duration: 0.4 }, 1.8)
      .from('.gs-quote', { autoAlpha: 0, x: 20, stagger: 0.12, duration: 0.35 }, 2.2)
      .from('.gs-review-pin', { autoAlpha: 0, scale: 0, stagger: 0.12, duration: 0.3, transformOrigin: 'center' }, 2.25)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, '>-0.1')
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="17" label="Case Study" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Nexus 6P
      </h2>
      <p className="gs-caption text-sm text-ink-100 leading-relaxed mb-4">
        Battery drain and bootloop defects collapsed a flagship.
      </p>

      <div className="grid grid-cols-[1fr_1fr] gap-4 flex-1 min-h-0">
        <div className="gs-chart bg-paper-100 border border-paper-300 rounded-xl p-4 overflow-hidden flex flex-col">
          <div className="flex gap-4 mb-2 text-sm text-ink-50">
            <span className="flex items-center gap-2"><span className="w-5 h-0.5 inline-block rounded" style={{ backgroundColor: AC.battery }} /> Battery</span>
            <span className="flex items-center gap-2"><span className="w-5 h-0.5 inline-block rounded" style={{ backgroundColor: AC.build }} /> Build</span>
            <span className="flex items-center gap-2"><span className="w-2.5 h-3.5 inline-block rounded-sm" style={{ backgroundColor: `${DET.cusum}40` }} /> CUSUM</span>
            <span className="flex items-center gap-2"><span className="w-2.5 h-3.5 inline-block rounded-sm border border-dashed" style={{ borderColor: DET.lstm }} /> LSTM</span>
          </div>
          <svg viewBox={`0 0 ${W} ${H}`} width="100%" className="overflow-visible flex-1">
            {[0, 0.25, 0.5, 0.75].map(v => {
              const y = Y0 + (1 - (v + 0.05) / 1.0) * (Y1 - Y0)
              return <line key={v} x1={X0} x2={X1} y1={y} y2={y} stroke={PAP.bone} strokeDasharray="3 3" />
            })}
            <path className="gs-batt-line" d={battPath} fill="none" stroke={AC.battery} strokeWidth={2} strokeLinecap="round" />
            <path className="gs-build-line" d={buildPath} fill="none" stroke={AC.build} strokeWidth={2} strokeLinecap="round" />
            {/* LSTM drift marker */}
            <rect className="gs-drift-marker" x={lstmX - 1} y={Y0} width={2} height={Y1 - Y0}
              fill={DET.lstm} opacity={0.25} style={{ transformOrigin: `${lstmX}px ${Y0}px` }} />
            <text className="gs-drift-marker" x={lstmX} y={Y0 - 4} textAnchor="middle" fontSize="10"
              fill={DET.lstm} fontFamily="JetBrains Mono" fontWeight="600">LSTM</text>
            {/* CUSUM drift marker */}
            <rect className="gs-drift-marker" x={cusumX - 1} y={Y0} width={2} height={Y1 - Y0}
              fill={DET.cusum} opacity={0.25} style={{ transformOrigin: `${cusumX}px ${Y0}px` }} />
            <text className="gs-drift-marker" x={cusumX} y={Y0 - 4} textAnchor="middle" fontSize="10"
              fill={DET.cusum} fontFamily="JetBrains Mono" fontWeight="600">CUSUM</text>
            {/* X labels: key date markers across the 24-month window (Jan 2016 – Dec 2017) */}
            {X_LABELS.map(([i, label]) => (
              <text key={i} x={X0 + (i / 23) * (X1 - X0)} y={Y1 + 14} textAnchor="middle" fontSize="11" fill={INK.dim} fontFamily="JetBrains Mono">
                {label}
              </text>
            ))}
            {/* Drift marker date labels */}
            <text className="gs-drift-marker" x={lstmX} y={Y1 + 14} textAnchor="end" fontSize="11"
              fill={DET.lstm} fontFamily="JetBrains Mono" fontWeight="600">2016-11</text>
            <text className="gs-drift-marker" x={cusumX} y={Y1 + 14} textAnchor="start" fontSize="11"
              fill={DET.cusum} fontFamily="JetBrains Mono" fontWeight="600">2017-03</text>
            {/* Review pointers: subtle color-matched pins anchoring each quote card to its timestamp */}
            {REVIEW_MARKERS.map(({ idx, color, series }) => {
              const x = X0 + (idx / 23) * (X1 - X0)
              const y = yAt(ASPECT_SERIES[series][idx])
              return (
                <g key={idx} className="gs-review-pin">
                  <circle cx={x} cy={y} r={4} fill={PAP.cream} stroke={color} strokeWidth={1.25} />
                  <circle cx={x} cy={y} r={1.75} fill={color} />
                </g>
              )
            })}
          </svg>
        </div>

        <div className="flex flex-col gap-2.5 justify-center min-h-0 overflow-hidden">
          <div className="gs-quote px-4 py-2.5 rounded-xl bg-paper-100 border border-paper-300">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-2 py-0.5 rounded-full text-[11px] text-white font-medium" style={{ backgroundColor: AC.battery }}>battery</span>
              <span className="text-[11px] text-ink-50">&#9733; &middot; 2017-05</span>
            </div>
            <p className="text-[12px] text-ink-200 leading-snug italic">
              &ldquo;Not even one year with the phone and now I am starting to have battery issues&hellip; the phone would unexpectedly shut down on its&rsquo; own regardless of remaining battery life&hellip;&rdquo;
            </p>
          </div>
          <div className="gs-quote px-4 py-2.5 rounded-xl bg-paper-100 border border-paper-300">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-2 py-0.5 rounded-full text-[11px] text-white font-medium" style={{ backgroundColor: AC.build }}>build</span>
              <span className="text-[11px] text-ink-50">&#9733; &middot; 2017-11</span>
            </div>
            <p className="text-[12px] text-ink-200 leading-snug italic">
              &ldquo;Huawei Phones were built with faulty main computer boards that crash to an unsolvable boot loop&hellip; this known fatal design flaw kicks in at 13 months&hellip;&rdquo;
            </p>
          </div>
          <div className="gs-quote px-4 py-2.5 rounded-xl bg-paper-100 border border-paper-300">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="px-2 py-0.5 rounded-full text-[11px] text-white font-medium" style={{ backgroundColor: AC.performance }}>performance</span>
              <span className="text-[11px] text-ink-50">&#9733; &middot; 2017-09</span>
            </div>
            <p className="text-[12px] text-ink-200 leading-snug italic">
              &ldquo;By far the worst device that I&rsquo;ve ever come across. The infamous bootloop issue that renders it a useless paperweight&hellip; a lemon of epic proportions.&rdquo;
            </p>
          </div>
        </div>
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-auto pt-4 justify-center">
        <span className="text-base text-ink-50">
          LSTM-AE flagged battery decay <span className="font-mono font-semibold text-ink-300">4 months</span> before CUSUM &middot; <span className="font-mono font-semibold text-ink-300">8 months</span> before the class-action lawsuit
        </span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Savings Waterfall
   ════════════════════════════════════════════════════════════════════ */
function SlideSavingsWaterfall() {
  const W = 540, H = 200
  const X0 = 55, X1 = 510
  const Y0 = 10, Y1 = 170
  const vMax = 7.5

  const yAt = (v: number) => Y1 - (v / vMax) * (Y1 - Y0)

  const barW = 58
  const xs = [90, 185, 280, 375, 470]

  const BARS = [
    { lines: ['Avoided', 'Recall Cost'], short: '+$4.2M', ref: '13', cumFrom: 0, cumTo: 4.2, pos: true, total: false },
    { lines: ['Reduced', 'Churn'], short: '+$1.8M', ref: '14', cumFrom: 4.2, cumTo: 6.0, pos: true, total: false },
    { lines: ['Engineering', 'Deflection'], short: '+$0.6M', ref: '15', cumFrom: 6.0, cumTo: 6.6, pos: true, total: false },
    { lines: ['Monitoring', 'Overhead'], short: '−$0.3M', ref: '16', cumFrom: 6.6, cumTo: 6.3, pos: false, total: false },
    { lines: ['Net Annual', 'Value'], short: '$6.3M', ref: '', cumFrom: 0, cumTo: 6.3, pos: true, total: true },
  ]

  const bridges = [
    { x1: xs[0] + barW / 2, x2: xs[1] - barW / 2, y: yAt(4.2) },
    { x1: xs[1] + barW / 2, x2: xs[2] - barW / 2, y: yAt(6.0) },
    { x1: xs[2] + barW / 2, x2: xs[3] - barW / 2, y: yAt(6.6) },
    { x1: xs[3] + barW / 2, x2: xs[4] - barW / 2, y: yAt(6.3) },
  ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-chart', { autoAlpha: 0, duration: 0.3 }, 0.45)
      .from('.gs-wbar', { autoAlpha: 0, y: 8, stagger: 0.12, duration: 0.4 }, 0.6)
      .from('.gs-wlabel', { autoAlpha: 0, y: -4, stagger: 0.12, duration: 0.3 }, 0.72)
      .from('.gs-bridge', { autoAlpha: 0, stagger: 0.1, duration: 0.25 }, 0.85)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="19" label="Business Value" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Cost <span className="italic text-ink-100">Savings</span>
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-4">
        A waterfall from drift cost to net annual savings (i.e., 10-product monitor, per year).
        <span className="block text-[11px] text-ink-50/80 mt-1 font-mono">
          Illustrative order-of-magnitude &middot; scope: 10 flagship products &middot; assumes ~2% per-product recall risk [13] and ~5% churn uplift recovery [14] &middot; human review before business action
        </span>
      </p>

      <div className="gs-chart relative flex-1 bg-paper-100 border border-paper-300 rounded-xl p-3 overflow-hidden">
        <svg viewBox={`0 0 ${W} ${H + 28}`} width="100%" className="overflow-visible">
          {/* Y-axis grid */}
          {[0, 2, 4, 6].map(v => (
            <g key={v}>
              <line x1={X0} x2={X1} y1={yAt(v)} y2={yAt(v)} stroke={PAP.bone} strokeDasharray="3 3" />
              <text x={X0 - 6} y={yAt(v) + 3.5} textAnchor="end" fontSize="9" fill={INK.dim} fontFamily="JetBrains Mono">${v}M</text>
            </g>
          ))}
          <line x1={X0} x2={X1} y1={Y1} y2={Y1} stroke={INK.med} strokeWidth={0.5} />

          {/* Bars */}
          {BARS.map((b, i) => {
            const cx = xs[i]
            const bx = cx - barW / 2
            const yTop = yAt(b.total ? b.cumTo : Math.max(b.cumFrom, b.cumTo))
            const yBot = yAt(b.total ? 0 : Math.min(b.cumFrom, b.cumTo))
            const bh = yBot - yTop
            const color = b.total ? INK.deep : b.pos ? SIG.teal : SIG.red
            return (
              <g key={i}>
                <rect className="gs-wbar" x={bx} y={yTop} width={barW} height={bh}
                  fill={color} opacity={b.total ? 0.88 : 0.70} rx={3} />
                <text className="gs-wlabel" x={cx} y={yTop - 5}
                  textAnchor="middle" fontSize="10" fontWeight="700"
                  fill={color} fontFamily="JetBrains Mono">
                  {b.short}
                </text>
                {b.lines.map((line, j) => (
                  <text key={j} x={cx} y={Y1 + 13 + j * 10}
                    textAnchor="middle" fontSize="7.5" fill={INK.dim} fontFamily="system-ui">
                    {line}
                    {b.ref && j === b.lines.length - 1 && (
                      <tspan dy="-2" fontSize="6" fontWeight="600" fill={INK.med}>[{b.ref}]</tspan>
                    )}
                  </text>
                ))}
              </g>
            )
          })}

          {/* Bridge lines */}
          {bridges.map((ln, i) => (
            <line key={i} className="gs-bridge"
              x1={ln.x1} x2={ln.x2} y1={ln.y} y2={ln.y}
              stroke={INK.dim} strokeDasharray="3 2" strokeWidth={0.75} opacity={0.45} />
          ))}
        </svg>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Detection Delay
   Source: results/drift/metrics.json (n=200 synthetic cases). Mean delay
   is months from injected drift onset to first firing (conditional on hit).
   ════════════════════════════════════════════════════════════════════ */
function SlideDetectionDelayCost() {
  const W = 560, H_VIEW = 205
  const X0 = 140, X1 = 510
  const Y0 = 20, Y1 = 155
  const tMax = 1.75  // months; Star Rating at 1.50 sits inside the axis

  const ROWS = [
    { name: 'CUSUM', delay: 1.11, recall: 0.570, color: DET.cusum },
    { name: 'LSTM-AE', delay: 0.69, recall: 0.335, color: DET.lstm },
    { name: 'PELT', delay: 0.65, recall: 0.155, color: INK.med },
    { name: 'Star Rating', delay: 1.50, recall: 0.010, color: PAP.stone },
  ]

  const xAt = (t: number) => X0 + (t / tMax) * (X1 - X0)
  const rowY = (i: number) => Y0 + 22 + i * 28

  /* Recall → dot radius (sqrt so area is proportional to recall). */
  const rAt = (recall: number) => 4 + Math.sqrt(recall) * 10

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-chart', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.45)
      .from('.gs-onset', { autoAlpha: 0, duration: 0.4 }, 0.8)
      .from('.gs-row-label', { autoAlpha: 0, x: -8, stagger: 0.1, duration: 0.35 }, 1.0)
      .from('.gs-row-bar', { scaleX: 0, transformOrigin: 'left center', stagger: 0.1, duration: 0.5, ease: 'power2.out' }, 1.15)
      .from('.gs-row-dot', { autoAlpha: 0, scale: 0, stagger: 0.1, duration: 0.35, transformOrigin: 'center' }, 1.5)
      .from('.gs-row-stat', { autoAlpha: 0, x: -6, stagger: 0.1, duration: 0.35 }, 1.65)
      .from('.gs-stat', { autoAlpha: 0, y: 10, stagger: 0.1, duration: 0.3 }, 2.0)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="20" label="Detection Delay" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Fast is not the same as <span className="italic text-ink-100">right.</span>
      </h2>
      <p className="gs-caption text-lg text-ink-100 leading-relaxed mb-4">
        Mean detection delay and recall across 200 synthetic drift cases; lower is faster, larger dots catch more events.
      </p>

      <div className="gs-chart relative flex-1 bg-paper-100 border border-paper-300 rounded-xl p-4">
        <svg viewBox={`0 0 ${W} ${H_VIEW}`} width="100%" height="100%">
          {/* X-axis gridlines + tick labels */}
          {[0, 0.5, 1.0, 1.5].map(t => (
            <g key={t}>
              <line x1={xAt(t)} x2={xAt(t)} y1={Y0} y2={Y1} stroke={PAP.bone} strokeDasharray="3 3" />
              <text x={xAt(t)} y={Y1 + 14} textAnchor="middle" fontSize="8.5" fill={INK.dim} fontFamily="JetBrains Mono">
                {t === 0 ? '0 mo' : `${t.toFixed(1)} mo`}
              </text>
            </g>
          ))}

          {/* Drift-onset reference line (t = 0) */}
          <g className="gs-onset">
            <line x1={xAt(0)} x2={xAt(0)} y1={Y0 - 4} y2={Y1 + 4}
              stroke={INK.deep} strokeWidth={1} />
            <text x={xAt(0)} y={Y0 - 8} textAnchor="middle" fontSize="8" fontWeight="600"
              fill={INK.deep} fontFamily="JetBrains Mono">drift onset</text>
          </g>

          {/* One row per detector: name | bar | dot (sized by recall) | recall + delay */}
          {ROWS.map((r, i) => {
            const y = rowY(i)
            const x0 = xAt(0)
            const x1 = xAt(r.delay)
            const dotR = rAt(r.recall)
            return (
              <g key={r.name}>
                <text className="gs-row-label"
                  x={X0 - 12} y={y + 3.5} textAnchor="end"
                  fontSize="11" fontWeight="600"
                  fill={INK.deep} fontFamily="JetBrains Mono">
                  {r.name}
                </text>

                <line className="gs-row-bar"
                  x1={x0} x2={x1} y1={y} y2={y}
                  stroke={r.color} strokeWidth={2.5} strokeLinecap="round" opacity={0.75} />

                <circle className="gs-row-dot"
                  cx={x1} cy={y} r={dotR}
                  fill={r.color} opacity={0.9} />

                <text className="gs-row-stat"
                  x={x1 + dotR + 8} y={y - 1} textAnchor="start"
                  fontSize="9.5" fontWeight="700"
                  fill={r.color} fontFamily="JetBrains Mono">
                  {(r.recall * 100).toFixed(1)}% recall
                </text>
                <text className="gs-row-stat"
                  x={x1 + dotR + 8} y={y + 10} textAnchor="start"
                  fontSize="8.5"
                  fill={INK.med} fontFamily="JetBrains Mono">
                  {r.delay.toFixed(2)} mo delay
                </text>
              </g>
            )
          })}

        </svg>
      </div>

      {/* Drift-injection protocol mini-panel: shows the 4 drift shapes driving the n=200 benchmark */}
      <div className="gs-chart mt-2 bg-paper-100 border border-paper-300 rounded-xl px-4 py-2">
        <div className="flex items-center gap-3">
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-50 shrink-0">
            Injected drift shapes &middot; n=200
          </span>
          <div className="flex-1 grid grid-cols-4 gap-3">
            {[
              { name: 'Step', desc: 'sudden \u22122.0', path: 'M0,10 L22,10 L22,22 L44,22' },
              { name: 'Gradual', desc: '\u22120.5/mo \u00d7 4', path: 'M0,10 L11,12 L22,16 L33,20 L44,22' },
              { name: 'Transient', desc: '1-month spike', path: 'M0,10 L18,10 L22,22 L26,10 L44,10' },
              { name: 'Multi', desc: '2\u20133 aspects', path: 'M0,10 L22,10 L22,20 L44,20 M0,14 L26,14 L26,22 L44,22' },
            ].map(d => (
              <div key={d.name} className="flex items-center gap-1.5 min-w-0">
                <svg width={44} height={24} viewBox="0 0 44 24" className="shrink-0">
                  <line x1={0} x2={44} y1={10} y2={10} stroke={INK.dim} strokeDasharray="2 2" strokeWidth={0.5} opacity={0.4} />
                  <path d={d.path} fill="none" stroke={SIG.red} strokeWidth={1.4} strokeLinecap="round" strokeLinejoin="round" />
                </svg>
                <div className="min-w-0 leading-tight">
                  <div className="font-mono text-[10px] font-semibold text-ink-300">{d.name}</div>
                  <div className="text-[9px] text-ink-50 truncate">{d.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="gs-stat flex items-baseline gap-3 mt-2 pt-2">
        <span className="text-sm text-ink-50">
          <span className="font-mono font-semibold" style={{ color: DET.cusum }}>CUSUM</span> catches the most (<span className="font-mono font-semibold text-ink-300">57.0%</span>) but waits longer (<span className="font-mono font-semibold text-ink-300">1.11 mo</span>)
        </span>
        <span className="text-sm text-ink-50 ml-6">
          <span className="font-mono font-semibold" style={{ color: DET.lstm }}>LSTM-AE</span> fires faster when it hits: <span className="font-mono font-semibold text-ink-300">33.5%</span> at <span className="font-mono font-semibold text-ink-300">0.69 mo</span>
        </span>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Sensitivity Analysis
   Source: 03_drift_detection.ipynb cells 15, 19, 23, 34.
   Real data only: quantile sweep q in {0.90, 0.92, 0.95, 0.97}, CUSUM h in
   {3, 4, 5}, PELT beta in {1, 2, 3, 5} (subset), and cross-seed val losses.
   ════════════════════════════════════════════════════════════════════ */
function SlideSensitivity() {
  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-block', { autoAlpha: 0, y: 18, stagger: 0.1, duration: 0.45 }, 0.45)
      .from('.gs-row', { autoAlpha: 0, x: -6, stagger: 0.04, duration: 0.28 }, 0.7)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.3 }, 1.3)
  })

  // LSTM-AE threshold quantile sweep (cell 23)
  const LSTM_SWEEP = [
    { q: '0.90', thr: 2.197, recall: 0.340, f1: 0.494, chosen: true },
    { q: '0.92', thr: 2.449, recall: 0.320, f1: 0.475 },
    { q: '0.95', thr: 2.838, recall: 0.220, f1: 0.357 },
    { q: '0.97', thr: 3.380, recall: 0.140, f1: 0.245 },
  ]

  // CUSUM h sweep (cell 15)
  const CUSUM_SWEEP = [
    { h: 3.0, recall: 0.570, hits: '114/200', chosen: true },
    { h: 4.0, recall: 0.430, hits: '86/200' },
    { h: 5.0, recall: 0.325, hits: '65/200' },
  ]

  // Architecture decisions (lstm_ae.py, notebook cell 3)
  const ARCH = [
    { param: 'seq_len', value: '6', why: 'Matches Phase 2 reference-window normalisation' },
    { param: 'hidden', value: '32', why: 'Small model \u2192 tight fit to normal patterns' },
    { param: 'latent', value: '8', why: 'Bottleneck forces compression; drift spikes error' },
    { param: 'dropout', value: '0.1', why: 'Small model + weight-decay + early stopping already regularise' },
  ]


  const lstmChosen = LSTM_SWEEP.find(r => r.chosen)!
  const cusumChosen = CUSUM_SWEEP.find(r => r.chosen)!

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="21" label="Tuning" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Sensitivity Analysis
      </h2>
      <p className="gs-caption text-sm text-ink-100 leading-relaxed mb-3">
        Thresholds swept on the n=200 synthetic benchmark; architecture fixed via multi-model consensus and validated across 3 random seeds.
      </p>

      <div className="flex-1 grid grid-cols-2 gap-3 min-h-0">
        {/* LEFT: threshold sweeps */}
        <div className="flex flex-col gap-3 min-h-0">

          {/* LSTM-AE sweep */}
          <div className="gs-block bg-paper-100 border border-paper-300 rounded-xl p-3.5 flex flex-col">
            <div className="flex items-baseline justify-between mb-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-50">LSTM-AE threshold sweep</span>
              <span className="font-mono text-[9px] text-ink-50">quantile of train recon errors</span>
            </div>

            {/* Chosen hero row */}
            <div
              className="gs-row flex items-center justify-between rounded-lg px-3 py-2 mb-2"
              style={{ backgroundColor: `${DET.lstm}18` }}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm font-bold text-ink-300">q = {lstmChosen.q}</span>
                <span className="font-mono text-[9px] px-1.5 py-0.5 rounded font-semibold" style={{ backgroundColor: `${DET.lstm}30`, color: DET.lstm }}>✓ chosen</span>
              </div>
              <div className="flex items-baseline gap-3">
                <div className="text-right">
                  <div className="font-mono text-lg font-bold text-ink-300">{(lstmChosen.recall * 100).toFixed(1)}%</div>
                  <div className="font-mono text-[9px] text-ink-50">recall</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm font-semibold text-ink-200">{lstmChosen.f1.toFixed(3)}</div>
                  <div className="font-mono text-[9px] text-ink-50">F1</div>
                </div>
              </div>
            </div>

            {/* Dimmed comparison rows */}
            <div className="grid grid-cols-[1fr_1fr_1fr_1fr] gap-1 font-mono text-[9px] uppercase tracking-wider text-ink-50 pb-0.5 border-b border-paper-300 px-1">
              <span>q</span><span className="text-right">thr</span><span className="text-right">recall</span><span className="text-right">F1</span>
            </div>
            {LSTM_SWEEP.filter(r => !r.chosen).map(r => (
              <div key={r.q} className="gs-row grid grid-cols-[1fr_1fr_1fr_1fr] gap-1 font-mono text-[10px] py-0.5 text-ink-50 px-1">
                <span className="tabular-nums">{r.q}</span>
                <span className="text-right tabular-nums">{r.thr.toFixed(3)}</span>
                <span className="text-right tabular-nums">{(r.recall * 100).toFixed(1)}%</span>
                <span className="text-right tabular-nums">{r.f1.toFixed(3)}</span>
              </div>
            ))}
            <p className="text-[9px] text-ink-50/80 mt-2 italic leading-tight">
              Window-level recall during threshold selection on synthetic validation; the final event-level recall after product-level fusion is 33.5% (shown in the operating-points callout below).
            </p>
          </div>

          {/* CUSUM sweep */}
          <div className="gs-block bg-paper-100 border border-paper-300 rounded-xl p-3.5 flex flex-col">
            <div className="flex items-baseline justify-between mb-2">
              <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-50">CUSUM threshold (h) sweep</span>
              <span className="font-mono text-[9px] text-ink-50">k = 0.5, cooldown = 2</span>
            </div>

            {/* Chosen hero row */}
            <div
              className="gs-row flex items-center justify-between rounded-lg px-3 py-2 mb-2"
              style={{ backgroundColor: `${DET.cusum}18` }}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-sm font-bold text-ink-300">h = {cusumChosen.h.toFixed(1)}</span>
                <span className="font-mono text-[9px] px-1.5 py-0.5 rounded font-semibold" style={{ backgroundColor: `${DET.cusum}30`, color: DET.cusum }}>✓ chosen</span>
              </div>
              <div className="flex items-baseline gap-3">
                <div className="text-right">
                  <div className="font-mono text-lg font-bold text-ink-300">{(cusumChosen.recall * 100).toFixed(1)}%</div>
                  <div className="font-mono text-[9px] text-ink-50">recall</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm font-semibold text-ink-200">{cusumChosen.hits}</div>
                  <div className="font-mono text-[9px] text-ink-50">hits / 200</div>
                </div>
              </div>
            </div>

            {/* Dimmed comparison rows */}
            <div className="grid grid-cols-[1fr_1fr_1.3fr] gap-1 font-mono text-[9px] uppercase tracking-wider text-ink-50 pb-0.5 border-b border-paper-300 px-1">
              <span>h</span><span className="text-right">recall</span><span className="text-right">hits / 200</span>
            </div>
            {CUSUM_SWEEP.filter(r => !r.chosen).map(r => (
              <div key={r.h} className="gs-row grid grid-cols-[1fr_1fr_1.3fr] gap-1 font-mono text-[10px] py-0.5 text-ink-50 px-1">
                <span className="tabular-nums">{r.h.toFixed(1)}</span>
                <span className="text-right tabular-nums">{(r.recall * 100).toFixed(1)}%</span>
                <span className="text-right tabular-nums">{r.hits}</span>
              </div>
            ))}
            <p className="text-[9px] text-ink-50/80 mt-2 italic leading-tight">
              PELT<span className="font-mono text-[8px] font-semibold text-ink-50 align-super ml-0.5">[4]</span> &beta; sweep &#123;1, 2, 3, 5&#125; on a 20-case subset capped at 2/20 recall across all settings &mdash; omitted from main panel; confirms PELT&rsquo;s weakness is structural, not a tuning artifact.
            </p>
          </div>
        </div>

        {/* RIGHT: architecture + seeds (compact) + operating points (prominent) */}
        <div className="flex flex-col gap-3 min-h-0">

          {/* Architecture + seeds — combined compact panel */}
          <div className="gs-block bg-paper-100 border border-paper-300 rounded-xl p-3.5">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-50 mb-2">
              LSTM-AE architecture &middot; locked before sweep
            </div>
            <div className="flex flex-wrap gap-1.5 mb-2">
              {ARCH.map(r => (
                <span key={r.param} className="gs-row font-mono text-xs bg-paper-200 border border-paper-300 px-2 py-0.5 rounded text-ink-200">
                  {r.param}<span className="text-ink-50">=</span>{r.value}
                </span>
              ))}
            </div>
            <p className="text-[10px] text-ink-50 leading-snug">
              seq_len matches Phase 2 normalisation &middot; hidden/latent bottleneck forces tight fit &middot; dropout=0.1 regularises alongside masking
            </p>
            <div className="mt-2 pt-2 border-t border-paper-300 flex items-center gap-1.5">
              <span className="font-mono text-[9px] uppercase tracking-wider text-ink-50">Seed stability</span>
              <span className="font-mono text-xs text-ink-200">sample &sigma;(val loss) = 0.009</span>
              <span className="text-ink-50 text-[9px]">&middot; seeds 42&ndash;44 all converge to ~0.72</span>
            </div>
          </div>

          {/* Operating points — prominent */}
          <div className="gs-block bg-paper-100 border border-paper-300 rounded-xl p-3.5 flex-1 flex flex-col justify-center">
            <div className="font-mono text-[10px] uppercase tracking-[0.16em] text-ink-50 mb-3">Latency &ndash; recall operating points</div>
            <div className="flex items-stretch gap-3">
              <div className="flex-1 flex flex-col items-center gap-1 p-3 rounded-lg" style={{ backgroundColor: `${DET.cusum}10`, border: `1px solid ${DET.cusum}25` }}>
                <span className="font-mono text-xs font-semibold" style={{ color: DET.cusum }}>CUSUM</span>
                <span className="font-mono text-2xl font-bold text-ink-300">57.0%</span>
                <span className="font-mono text-sm text-ink-200">1.11 mo</span>
                <span className="text-[10px] text-ink-50 text-center mt-1">accumulates &rarr; finds more</span>
              </div>
              <div className="flex items-center text-ink-50 text-sm">vs</div>
              <div className="flex-1 flex flex-col items-center gap-1 p-3 rounded-lg" style={{ backgroundColor: `${DET.lstm}10`, border: `1px solid ${DET.lstm}25` }}>
                <span className="font-mono text-xs font-semibold" style={{ color: DET.lstm }}>LSTM-AE</span>
                <span className="font-mono text-2xl font-bold text-ink-300">33.5%</span>
                <span className="font-mono text-sm text-ink-200">0.69 mo</span>
                <span className="text-[10px] text-ink-50 text-center mt-1">single window &rarr; fires earlier</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Deployment Recommendations
   ════════════════════════════════════════════════════════════════════ */
function SlideDeployment() {
  const RECS: Array<{
    num: string; title: string; desc: string; descRef?: string;
    risk: string; integration: string; integrationRef?: string; icon: string;
  }> = [
    {
      num: '01', title: 'CUSUM as primary monitor',
      desc: 'Highest recall (57.0%), zero training data, runs on raw z-scores. Deploy per-aspect on all 335 products.',
      descRef: '5',
      risk: 'False-alarm fatigue \u2192 confidence score',
      integration: 'Ships over REST into existing IntelliSys client dashboards with no new ML runtime to host.',
      icon: DET.cusum
    },
    {
      num: '02', title: 'LSTM-AE as early warning',
      desc: 'Faster detection when it fires (0.69 mo vs CUSUM 1.11 mo). AND-gated with CUSUM to reduce false positives.',
      risk: 'Threshold drift \u2192 monthly recalibration on rolling 12-month window.',
      integration: 'ONNX export keeps it framework-agnostic for edge or cloud deployments.',
      integrationRef: '12',
      icon: DET.lstm
    },
    {
      num: '03', title: 'Human-in-the-loop always',
      desc: 'Weakly supervised system as all detections require human review before business action.',
      risk: 'Reviewer bandwidth \u2192 auto-cluster similar alerts; escalate only top-severity.',
      integration: 'Alerts land in the same queue as sensor anomalies.',
      icon: SIG.red
    },
  ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-tier', { autoAlpha: 0, y: 16, stagger: 0.18, duration: 0.4 }, 0.45)
      .from('.gs-rec', { autoAlpha: 0, x: 16, stagger: 0.12, duration: 0.4 }, 0.55)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="22" label="Recommendations" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-6">
        Deployment <span className="italic text-ink-100">Architecture</span>
      </h2>

      <div className="flex gap-4 flex-1 min-h-0">
        {/* LEFT: Tier diagram */}
        <div className="flex flex-col gap-1.5 min-h-0" style={{ width: '54%' }}>
          {/* Input source */}
          <div className="gs-tier rounded-lg px-3 py-2 text-center shrink-0"
            style={{ background: PAP.bone, border: `1px solid ${PAP.stone}` }}>
            <p className="text-xs font-semibold text-ink-300">Reviews + product metadata</p>
            <p className="font-mono text-[9px] text-ink-50">312,100 reviews · 335 products</p>
          </div>

          {/* Arrow */}
          <div className="flex items-center justify-center shrink-0 py-0.5">
            <div className="flex flex-col items-center gap-0.5">
              <div style={{ width: 1, height: 8, background: INK.dim }} />
              <svg width="8" height="5" viewBox="0 0 8 5" fill="none"><path d="M4 5L0 0h8z" fill={INK.dim} /></svg>
            </div>
          </div>

          {/* Cloud tier */}
          <div className="gs-tier rounded-xl p-3 shrink-0"
            style={{ background: `${SIG.red}0C`, border: `2px solid ${SIG.red}55` }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-[9px] font-bold uppercase tracking-[0.2em]" style={{ color: SIG.red }}>Cloud</span>
              <span className="font-mono text-[8px]" style={{ color: INK.dim }}>GPU · batch</span>
            </div>
            <div className="rounded-lg px-3 py-2 bg-paper-100 border border-paper-300">
              <p className="text-xs font-semibold text-ink-300">ABSA extraction</p>
              <p className="font-mono text-[9px] text-ink-50">PyABSA<span className="font-semibold text-ink-200 align-super ml-0.5">[2]</span> DeBERTa-v3 · ~184M params</p>
            </div>
          </div>

          {/* Arrow + label */}
          <div className="flex items-center justify-center gap-1.5 shrink-0 py-0.5">
            <div className="flex flex-col items-center gap-0.5">
              <div style={{ width: 1, height: 6, background: INK.dim }} />
              <svg width="8" height="5" viewBox="0 0 8 5" fill="none"><path d="M4 5L0 0h8z" fill={INK.dim} /></svg>
            </div>
            <span className="font-mono text-[9px]" style={{ color: INK.dim }}>Aspect scores · monthly</span>
          </div>

          {/* On-premise tier */}
          <div className="gs-tier rounded-xl p-3 shrink-0"
            style={{ background: `${DET.lstm}12`, border: `2px solid ${DET.lstm}60` }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-[9px] font-bold uppercase tracking-[0.2em]" style={{ color: DET.lstm }}>On-premise</span>
              <span className="font-mono text-[8px]" style={{ color: INK.dim }}>CPU · ONNX</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <div className="rounded-lg px-2 py-1.5 bg-paper-100 border border-paper-300">
                <p className="text-xs font-semibold text-ink-300">CUSUM<span className="font-mono text-[9px] font-semibold text-ink-50 align-super ml-0.5">[5]</span></p>
                <p className="font-mono text-[9px] text-ink-50">57% recall · zero training</p>
              </div>
              <div className="rounded-lg px-2 py-1.5 bg-paper-100 border border-paper-300">
                <p className="text-xs font-semibold text-ink-300">LSTM-AE</p>
                <p className="font-mono text-[9px] text-ink-50">0.69 mo delay · ~11 KB</p>
              </div>
            </div>
          </div>

          {/* Arrow + label */}
          <div className="flex items-center justify-center gap-1.5 shrink-0 py-0.5">
            <div className="flex flex-col items-center gap-0.5">
              <div style={{ width: 1, height: 6, background: INK.dim }} />
              <svg width="8" height="5" viewBox="0 0 8 5" fill="none"><path d="M4 5L0 0h8z" fill={INK.dim} /></svg>
            </div>
            <span className="font-mono text-[9px] italic" style={{ color: INK.dim }}>Distilled<span className="font-semibold align-super ml-0.5">[10]</span> · quarterly (opt.)</span>
          </div>

          {/* Edge tier */}
          <div className="gs-tier rounded-xl p-3 flex-1 min-h-0"
            style={{ background: `${AC.connectivity}0C`, border: `2px dashed ${AC.connectivity}60` }}>
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-[9px] font-bold uppercase tracking-[0.2em]" style={{ color: AC.connectivity }}>Edge</span>
              <span className="font-mono text-[8px]" style={{ color: INK.dim }}>Cortex-M · INT8 · opt.</span>
            </div>
            <div className="grid grid-cols-2 gap-1.5">
              <div className="rounded-lg px-2 py-1.5 bg-paper-100 border border-paper-300">
                <p className="text-xs font-semibold text-ink-300">Quantized<span className="font-mono text-[9px] font-semibold text-ink-50 align-super ml-0.5">[9]</span> LSTM-AE</p>
                <p className="font-mono text-[9px] text-ink-50">~3 KB INT8</p>
              </div>
              <div className="rounded-lg px-2 py-1.5 bg-paper-100 border border-paper-300">
                <p className="text-xs font-semibold text-ink-300">Embedded CUSUM<span className="font-mono text-[9px] font-semibold text-ink-50 align-super ml-0.5">[5]</span></p>
                <p className="font-mono text-[9px] text-ink-50">Pure arithmetic</p>
              </div>
            </div>
          </div>
        </div>

        {/* RIGHT: Recommendation cards */}
        <div className="flex flex-col gap-2 flex-1 min-h-0">
          {RECS.map(r => (
            <div key={r.num} className="gs-rec bg-paper-100 border border-paper-300 rounded-xl p-3 flex gap-2.5 flex-1 min-h-0 overflow-hidden">
              <div className="flex-shrink-0 w-7 h-7 rounded-md flex items-center justify-center mt-0.5"
                style={{ backgroundColor: `${r.icon}15`, border: `1px solid ${r.icon}30` }}>
                <span className="font-mono text-[10px] font-semibold" style={{ color: r.icon }}>{r.num}</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-ink-300 mb-1">{r.title}</p>
                <p className="text-[10px] text-ink-100 leading-snug mb-1.5">
                  {r.desc}
                  {r.descRef && <span className="font-mono text-[9px] font-semibold text-ink-50 align-super ml-0.5">[{r.descRef}]</span>}
                </p>
                <div className="flex items-start gap-1 mb-1">
                  <span className="font-mono text-[8px] uppercase tracking-[0.15em] font-semibold text-signal-red shrink-0 pt-0.5">Risk</span>
                  <p className="text-[10px] text-ink-100 leading-snug">{r.risk}</p>
                </div>
                <div className="flex items-start gap-1">
                  <span className="font-mono text-[8px] uppercase tracking-[0.15em] font-semibold text-signal-teal shrink-0 pt-0.5">Fit</span>
                  <p className="text-[10px] text-ink-100 leading-snug">
                    {r.integration}
                    {r.integrationRef && <span className="font-mono text-[9px] font-semibold text-ink-50 align-super ml-0.5">[{r.integrationRef}]</span>}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Production ML
   ════════════════════════════════════════════════════════════════════ */
function SlideProductionML() {
  const SCENARIOS: Array<{
    num: string; tag: string; title: string; accent: string;
    body: string; tradeoff: string;
    chips: Array<{ label: string; ref: string }>; recommended?: boolean;
  }> = [
      {
        num: '01', tag: '', title: 'Full Cloud', accent: SIG.red,
        body: 'Monthly reviews \u2192 S3 \u2192 GPU batch ABSA \u2192 feature store \u2192 CUSUM/LSTM-AE \u2192 REST to dashboards. Spin up T4/L4 for a few hours/month, then down.',
        tradeoff: 'Raw review data leaves client perimeter',
        chips: [{ label: 'ONNX Export', ref: '12' }],
      },
      {
        num: '02', tag: '', title: 'On-premise', accent: SIG.teal,
        body: 'Nothing leaves the network. CUSUM is pure Python/numpy. LSTM-AE via ONNX Runtime on existing CPU box. ABSA needs on-premesis GPU or TinyBERT distillation',
        tradeoff: 'Currently need on-premesis GPU; distillation sprint unlocks CPU-only',
        chips: [{ label: 'ONNX Export', ref: '12' }, { label: 'Distillation (future)', ref: '10' }],
      },
      {
        num: '03', tag: '', title: 'Edge', accent: AC.connectivity,
        body: 'Push detection to Cortex-M: quantized LSTM-AE (~3\u202fKB INT8) + embedded CUSUM over pre-computed aspect scores. ABSA stays off-device.',
        tradeoff: 'Only needed if IntelliSys embeds watchers into client gateways/devices',
        chips: [{ label: 'Quantization', ref: '9' }, { label: 'Pruning (future)', ref: '11' }],
      },
      {
        num: '04', tag: '', title: 'Hybrid', accent: DET.lstm, recommended: true,
        body: "ABSA in cloud (GPU needed there anyway). CUSUM + LSTM-AE on-prem via ONNX on IntelliSys\u2019s existing CPU infra. Edge behind dashed line. Monthly aspect-score handoff keeps data-transfer surface tiny.",
        tradeoff: 'Recommended as it balances cost, data-residency, simplicity',
        chips: [{ label: 'ONNX Export', ref: '12' }, { label: 'Distillation (future)', ref: '10' }],
      },
    ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-rec', { autoAlpha: 0, y: 20, stagger: 0.12, duration: 0.45 }, 0.5)
      .from('.gs-chip', { autoAlpha: 0, y: 6, stagger: 0.035, duration: 0.3 }, 0.9)
      .from('.gs-stat', { autoAlpha: 0, y: 10, duration: 0.35 }, 1.2)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="23" label="Production ML" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-3">
        Deployment <span className="italic text-ink-100">Scenarios</span>
      </h2>
      <p className="gs-caption text-base text-ink-100 leading-relaxed mb-4">
        Four ways to split ABSA, CUSUM, and LSTM-AE across cloud, on-premise, and edge
      </p>

      <div className="flex-1 grid grid-cols-2 gap-3 auto-rows-fr min-h-0">
        {SCENARIOS.map(s => (
          <div key={s.num}
            className="gs-rec relative bg-paper-100 rounded-xl px-4 py-3.5 flex flex-col overflow-hidden"
            style={{ border: s.recommended ? `2px solid ${s.accent}70` : `1px solid ${PAP.stone}` }}>
            {s.recommended && (
              <div className="absolute top-0 right-0 px-2.5 py-0.5 rounded-bl-lg rounded-tr-xl"
                style={{ background: `${s.accent}20`, borderLeft: `1px solid ${s.accent}40`, borderBottom: `1px solid ${s.accent}40` }}>
                <span className="font-mono text-[8px] font-bold uppercase tracking-[0.2em]" style={{ color: s.accent }}>Recommended</span>
              </div>
            )}
            <div className="flex items-start justify-between gap-3 mb-2.5">
              <div className="flex items-baseline gap-2.5 min-w-0">
                <span className="font-mono text-sm font-semibold tabular-nums shrink-0" style={{ color: s.accent }}>{s.num}</span>
                <p className="text-sm font-semibold text-ink-300 leading-tight">{s.title}</p>
              </div>
              <span className="font-mono text-[9px] uppercase tracking-[0.22em] font-semibold shrink-0 pt-0.5"
                style={{ color: s.accent }}>
                {s.tag}
              </span>
            </div>

            <p className="text-xs text-ink-100 leading-relaxed mb-2.5 flex-1">{s.body}</p>

            <div className="flex items-start gap-1.5 mb-3">
              {/* <span className="font-mono text-[8px] uppercase tracking-[0.15em] font-semibold text-signal-red shrink-0 pt-0.5">Risk</span> */}
              <p className="text-[10px] text-ink-100 leading-snug">{s.tradeoff}</p>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {s.chips.map(c => (
                <span key={c.label}
                  className="gs-chip inline-flex items-center px-2 py-0.5 rounded-md text-[9px] font-mono font-semibold tracking-wide"
                  style={{ backgroundColor: `${s.accent}14`, color: s.accent, border: `1px solid ${s.accent}33` }}>
                  {c.label}<span className="font-semibold align-super ml-0.5 opacity-70">[{c.ref}]</span>
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <p className="gs-stat shrink-0 text-xs text-center text-ink-100 mt-3">
        <strong className="font-semibold" style={{ color: DET.lstm }}>Hybrid</strong> is the default answer &middot; <strong className="font-semibold" style={{ color: SIG.teal }}>On-prem</strong> forces the TinyBERT distillation<span className="font-mono text-[10px] font-semibold text-ink-50 align-super ml-0.5">[10]</span> sprint &middot; <strong className="font-semibold text-ink-200">Edge</strong> useful only at gateway scale
      </p>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Limitations & Ethics
   ════════════════════════════════════════════════════════════════════ */
function SlideLimitations() {
  const LIMITS: Array<{ title: string; desc: string; ref?: string }> = [
    { title: 'Sparse months', desc: 'Shrinkage helps but low-volume months remain noisy' },
    { title: 'Weakly supervised', desc: 'No ground-truth aspect labels; attribution is approximate', ref: '7,8' },
    { title: 'Threshold sensitivity', desc: 'LSTM-AE and CUSUM thresholds trade recall against precision', ref: '6' },
    { title: 'Historical only', desc: 'Trained on past data; concept drift may degrade over time' },
  ]

  const ETHICS: Array<{ title: string; risk: string; mit: string; ref?: string }> = [
    {
      title: 'Privacy (GDPR)',
      risk: 'Individual reviewer identification if outputs leak back to review-level.',
      mit: 'Processing performed on product \u00d7 aspect \u00d7 month aggregates; reviewer identifiers dropped before analysis. Any intermediate hash is an engineering convenience \u2014 hashed IDs remain personal data under GDPR Art\u202f4(5) and Recital\u202f26 and are not the privacy control.',
      ref: '17'
    },
    {
      title: 'Demographic bias',
      risk: 'Amazon reviewers skew tech-savvy, male, US-centric; signals may under-represent others.',
      mit: 'Verified vs unverified sentiment reported separately; bias disclosure on every alert.'
    },
    {
      title: 'Misuse / fake reviews',
      risk: 'Coordinated fake reviews could trigger false alerts or mask real drift.',
      mit: 'Volume + verified-ratio gate; surge-day flag excludes suspicious spikes.'
    },
    {
      title: 'Transparency',
      risk: 'Black-box alerts erode analyst trust and block action.',
      mit: 'Every alert shows aspect deltas + representative review quotes; token-level attributions via Layer Integrated Gradients.',
      ref: '7,8'
    },
    {
      title: 'Reliability / human-in-loop',
      risk: 'Auto-action on a weakly supervised signal could harm a supplier unfairly.',
      mit: 'All alerts require human review before business action; mitigates ML-system technical debt of feedback loops.',
      ref: '16'
    },
    {
      title: 'Environmental cost',
      risk: 'Heavy LLM inference at scale has a real carbon footprint.',
      mit: 'LSTM-AE runs CPU-only (hidden=32); PyABSA inference local and batched.'
    },
  ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-col-head', { autoAlpha: 0, y: 12, stagger: 0.1, duration: 0.3 }, 0.45)
      .from('.gs-item', { autoAlpha: 0, y: 14, stagger: 0.08, duration: 0.35 }, 0.6)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="24" label="Limitations & Ethics" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Reflection on Limitations & Ethics
      </h2>

      <div className="grid grid-cols-12 gap-4 flex-1 min-h-0 overflow-hidden">
        <div className="col-span-4 flex flex-col min-h-0">
          <p className="gs-col-head font-mono text-[10px] uppercase tracking-[0.15em] text-signal-red mb-2">Limitations</p>
          <div className="flex flex-col gap-2 flex-1 min-h-0">
            {LIMITS.map(l => (
              <div key={l.title} className="gs-item bg-paper-100 border border-paper-300 rounded-lg px-4 py-3 flex flex-col justify-center flex-1">
                <p className="text-xs font-semibold text-ink-300 mb-1">{l.title}</p>
                <p className="text-[11px] text-ink-100 leading-relaxed">
                  {l.desc}
                  {l.ref && <span className="font-mono text-[9px] font-semibold text-ink-50 align-super ml-0.5">[{l.ref}]</span>}
                </p>
              </div>
            ))}
          </div>
        </div>
        <div className="col-span-8 flex flex-col min-h-0">
          <p className="gs-col-head font-mono text-[10px] uppercase tracking-[0.15em] text-signal-teal mb-2">Ethics: risk &rarr; safeguard</p>
          <div className="grid grid-cols-2 gap-2 flex-1 min-h-0">
            {ETHICS.map(e => (
              <div key={e.title} className="gs-item bg-paper-100 border border-paper-300 rounded-lg px-4 py-3 flex flex-col justify-center">
                <p className="text-xs font-semibold text-ink-300 mb-2.5">{e.title}</p>
                <p className="text-[10px] text-ink-100 leading-relaxed mb-2"><span className="font-mono text-[9px] uppercase tracking-[0.15em] font-semibold text-signal-red mr-1">Risk</span>{e.risk}</p>
                <p className="text-[10px] text-ink-100 leading-relaxed"><span className="font-mono text-[9px] uppercase tracking-[0.15em] font-semibold text-signal-teal mr-1">Mit</span>{e.mit}{e.ref && <span className="font-mono text-[9px] font-semibold text-ink-50 align-super ml-0.5">[{e.ref}]</span>}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: References
   ════════════════════════════════════════════════════════════════════ */
function SlideReferences() {
  const REFS: Array<{ n: string; tag: string; cite: string; use: string }> = [
    {
      n: '1', tag: 'Dataset',
      cite: 'Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024). Bridging Language and Items for Retrieval and Recommendation. arXiv:2403.03952.',
      use: 'Amazon Reviews 2023, MIT licence; 1,501 smartphone products, 312,100 cleaned reviews.'
    },
    {
      n: '2', tag: 'ABSA',
      cite: 'Yang, H., Zhang, C. & Li, K. (2023). PyABSA: A Modularized Framework for Reproducible Aspect-Based Sentiment Analysis. arXiv:2208.01368.',
      use: 'FAST-LSA-T-V2 APC head, achieving 0.819 CV F1 on 195 gold triplets.'
    },
    {
      n: '3', tag: 'ABSA fallback',
      cite: 'Scaria, K., Gupta, H., Goyal, S., Sawant, S. A., Mishra, S., & Baral, C. (2024). InstructABSA: Instruction Learning for Aspect-Based Sentiment Analysis. Findings of NAACL 2024. arXiv:2302.08624.',
      use: 'Documented fallback if PyABSA fails domain transfer.'
    },
    {
      n: '4', tag: 'Change-point',
      cite: 'Killick, R., Fearnhead, P. & Eckley, I. (2012). Optimal Detection of Changepoints with a Linear Computational Cost. JASA 107(500), 1590\u20131598.',
      use: 'PELT (offline) statistical baseline via ruptures, CostRank.'
    },
    {
      n: '5', tag: 'Sequential',
      cite: 'Page, E. S. (1954). Continuous Inspection Schemes. Biometrika 41(1/2), 100\u2013115.',
      use: 'CUSUM (online) statistical baseline, two-sided, per-aspect.'
    },
    {
      n: '6', tag: 'CPD survey',
      cite: 'Truong, C., Oudre, L. & Vayatis, N. (2020). Selective Review of Offline Change Point Detection Methods. Signal Processing 167.',
      use: 'Justifies method choice and benchmarking protocol.'
    },
    {
      n: '7', tag: 'Attribution',
      cite: 'Sundararajan, M., Taly, A. & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. ICML 2017.',
      use: 'Layer Integrated Gradients for per-token attribution on the PyABSA encoder (slide 7).'
    },
    {
      n: '8', tag: 'Attribution library',
      cite: 'Kokhlikyan, N. et al. (2020). Captum: A Unified and Generic Model Interpretability Library for PyTorch. arXiv:2009.07896.',
      use: 'Captum LayerIntegratedGradients drives the token-level heatmaps on slide 7.'
    },
    {
      n: '9', tag: 'Quantization',
      cite: 'Jacob, B. et al. (2018). Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR.',
      use: 'INT8 yields 4\u00d7 model-size reduction, 2\u20134\u00d7 compute speedup at <1% accuracy loss.'
    },
    {
      n: '10', tag: 'Distillation',
      cite: 'Hinton, Vinyals & Dean (2015). Distilling the Knowledge in a Neural Network. arXiv:1503.02531; Jiao et al. (2020). TinyBERT. Findings of EMNLP.',
      use: '96.8% of BERT-base accuracy at 7.5\u00d7 smaller, 9.4\u00d7 faster inference.'
    },
    {
      n: '11', tag: 'Pruning',
      cite: 'Han et al. (2015). Learning both Weights and Connections for Efficient Neural Networks. NeurIPS; Narang et al. (2017). Exploring Sparsity in RNNs. ICLR.',
      use: 'Up to 90% LSTM sparsity with minor accuracy loss.'
    },
    {
      n: '12', tag: 'Serving',
      cite: 'Bai, Lu, Chen et al. (2019). Open Neural Network Exchange specification & ONNX Runtime. Microsoft. https://onnx.ai',
      use: 'Cross-framework, cross-platform model serving for deployment.'
    },
    {
      n: '13', tag: 'Recall cost',
      cite: 'Reuters (2017). Samsung Note 7 recall cost estimate US$5.3B; Stericycle Recall Index (2023), mean electronics recall US$10M\u2013100M.',
      use: 'Waterfall input: avoided recall cost, i.e., +$4.2M (slide 20).'
    },
    {
      n: '14', tag: 'Churn',
      cite: 'Reichheld, F. & Sasser, W. E. (1990). Zero Defections: Quality Comes to Services. Harvard Business Review; Bain & Co. (2001) follow-up.',
      use: 'Waterfall input: reduced churn, i.e., +$1.8M (slide 20). 5% retention lift \u2192 25\u201395% profit uplift.'
    },
    {
      n: '15', tag: 'Engineering cost',
      cite: 'Boehm, B. (1981). Software Engineering Economics. Prentice Hall; IBM Systems Sciences Institute, defects cost 6\u2013100\u00d7 more in production.',
      use: 'Waterfall input: engineering deflection, i.e., +$0.6M (slide 20).'
    },
    {
      n: '16', tag: 'ML ops cost',
      cite: 'Sculley, D. et al. (2015). Hidden Technical Debt in Machine Learning Systems. NeurIPS; Gartner MLOps cost benchmarks (2023).',
      use: 'Waterfall input: monitoring overhead, i.e., \u2212$0.3M (slide 20).'
    },
    {
      n: '17', tag: 'Privacy (GDPR)',
      cite: 'European Parliament & Council (2016). Regulation (EU) 2016/679 (GDPR), Art\u202f4(5) (definition of pseudonymisation) and Recital\u202f26 (hashed identifiers remain personal data). Official Journal of the EU, L 119/1.',
      use: 'Basis for privacy safeguard on slide 24: hashed reviewer IDs are not the privacy control; only aggregate-level outputs (product \u00d7 aspect \u00d7 month) are retained.'
    },
  ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-ref-item', { autoAlpha: 0, y: 10, stagger: 0.05, duration: 0.3 }, 0.45)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="27" label="References" />
      <p className="gs-caption text-sm text-ink-100 leading-relaxed mb-3">
        Dataset, ABSA, change-point detection and attribution references. Full BibTeX in <code className="font-mono text-xs">references.bib</code>.
      </p>

      <div className="flex-1 min-h-0 overflow-hidden">
        <ol className="flex flex-col gap-1">
          {REFS.map((r, i) => (
            <li key={i} className="gs-ref-item flex items-baseline gap-2.5 leading-snug">
              <span className="font-mono text-[11px] font-semibold text-ink-300 shrink-0 w-7 text-right tabular-nums">[{r.n}]</span>
              <span className="font-mono text-[9px] uppercase tracking-[0.15em] font-semibold text-signal-teal shrink-0 w-32">{r.tag}</span>
              <span className="text-[11px] text-ink-100 flex-1">{r.cite}</span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Team Contribution
   ════════════════════════════════════════════════════════════════════ */
function SlideTeam() {
  const TEAM: Array<{ name: string; role: string; areas: string[] }> = [
    { name: 'Alex Young', role: 'M1 Data lead', areas: ['EDA + cleaning (00)', 'Monthly aggregation (02)', 'Dashboard + slides'] },
    { name: 'Ghulam Ahmed', role: 'Modelling', areas: ['ABSA evaluation (01)', 'Gold-label validation'] },
    { name: 'Dilara Bayram', role: 'Modelling', areas: ['LSTM-AE architecture (03)', 'Threshold tuning'] },
    { name: 'Ubay Alshamali', role: 'Baselines', areas: ['PELT + CUSUM (03)', 'Synthetic benchmark design'] },
    { name: 'Adam Okeahalam', role: 'Explainability', areas: ['Aspect-delta ranking (04)', 'Case-study curation'] },
    { name: 'Nicholas Saremi', role: 'Business framing', areas: ['IntelliSys recommendations', 'Risk / ethics narrative'] },
  ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-caption', { autoAlpha: 0, y: 16, duration: 0.4 }, 0.3)
      .from('.gs-member', { autoAlpha: 0, y: 10, stagger: 0.07, duration: 0.35 }, 0.45)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="26" label="Team" />
      <h2 className="gs-heading font-display text-3xl md:text-4xl leading-tight mb-2">
        Team &amp; <span className="italic text-ink-100">contribution.</span>
      </h2>
      <p className="gs-caption text-sm text-ink-100 leading-relaxed mb-4">
        Every member contributed substantively. Peer-review scores filed on Moodle separately.
      </p>

      <div className="grid grid-cols-2 gap-2 flex-1 min-h-0 overflow-hidden">
        {TEAM.map(m => (
          <div key={m.name} className="gs-member bg-paper-100 border border-paper-300 rounded-lg px-3 py-2 flex flex-col">
            <div className="flex items-baseline justify-between mb-1">
              <p className="text-sm font-semibold text-ink-300">{m.name}</p>
              <span className="font-mono text-[9px] uppercase tracking-[0.15em] text-signal-teal">{m.role}</span>
            </div>
            <ul className="text-[11px] text-ink-100 leading-snug space-y-0.5">
              {m.areas.map(a => (
                <li key={a} className="flex gap-1.5"><span className="text-signal-teal">&bull;</span><span>{a}</span></li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   SLIDE: Conclusions
   ════════════════════════════════════════════════════════════════════ */
function SlideConclusions() {
  const TAKEAWAYS = [
    {
      num: '1', headline: 'Star ratings are not enough',
      detail: 'Caught only 1.0% of drift events. Aspect-level analysis is non-negotiable.'
    },
    {
      num: '2', headline: 'CUSUM leads, LSTM-AE complements',
      detail: '57.0% vs 33.5% recall, though LSTM-AE fires ~1.6\u00d7 faster (0.69 mo vs 1.11 mo) when it hits.'
    },
    {
      num: '3', headline: 'Right tool per layer: DL for text, statistics for signals',
      detail: 'PyABSA: +51.5\u202fpp F1 over TF-IDF+LR (DL correct at the feature layer). CUSUM: 57.0% vs 33.5% recall (classical statistics correct at the detection layer). LSTM-AE: complementary early-warning via orthogonal failure modes, not duplication.'
    },
    {
      num: '4', headline: 'Every alert is explainable',
      detail: 'Aspect deltas + review evidence ground every detection in customer language.'
    },
  ]

  const containerRef = useSlideTimeline((tl, _el) => {
    tl.from('.gs-label', { autoAlpha: 0, x: -10, duration: 0.3 }, 0.1)
      .from('.gs-heading', { autoAlpha: 0, y: 24, duration: 0.5 }, 0.15)
      .from('.gs-takeaway', { autoAlpha: 0, x: -20, stagger: 0.15, duration: 0.45 }, 0.4)
      .from('.gs-footer', { autoAlpha: 0, y: 10, duration: 0.3 }, 1.2)
  })

  return (
    <div ref={containerRef} className="flex-1 flex flex-col min-h-0">
      <SlideLabel num="25" label="Conclusions" />
      <h2 className="gs-heading font-display text-4xl md:text-5xl leading-tight mb-6">
        <span className="italic text-ink-100">Summary</span>
      </h2>

      <div className="space-y-4 flex-1 flex flex-col justify-center max-w-2xl mx-auto w-full">
        {TAKEAWAYS.map(t => (
          <div key={t.num} className="gs-takeaway flex items-start gap-5 p-5 bg-paper-100 border border-paper-300 rounded-xl">
            <span className="font-mono text-3xl font-semibold text-signal-teal flex-shrink-0 w-10 text-center">{t.num}</span>
            <div>
              <p className="text-lg font-medium text-ink-300 mb-1">{t.headline}</p>
              <p className="text-sm text-ink-100 leading-relaxed">{t.detail}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

/* ════════════════════════════════════════════════════════════════════
   MAIN TUTORIAL COMPONENT
   ════════════════════════════════════════════════════════════════════ */
export default function Tutorial({ onDismiss }: Props) {
  const [slide, setSlide] = useState(0)
  const [dir, setDir] = useState(0)

  const goNext = useCallback(() => {
    if (slide < TOTAL_SLIDES - 1) { setDir(1); setSlide(s => s + 1) }
  }, [slide])
  const goPrev = useCallback(() => {
    if (slide > 0) { setDir(-1); setSlide(s => s - 1) }
  }, [slide])

  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      if (e.key === 'ArrowRight' || e.key === ' ') goNext()
      else if (e.key === 'ArrowLeft') goPrev()
      else if (e.key === 'Escape') onDismiss()
      else if (e.key === 'Enter' && slide === TOTAL_SLIDES - 1) onDismiss()
    }
    window.addEventListener('keydown', h)
    return () => window.removeEventListener('keydown', h)
  }, [goNext, goPrev, onDismiss, slide])

  return (
    <motion.div
      className="fixed inset-0 z-[100] flex items-center justify-center"
      initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div className="absolute inset-0 bg-ink-300/95" />
      <button onClick={onDismiss} className="absolute top-6 right-8 z-10 text-paper-400 hover:text-paper-50 text-xs font-mono transition-colors">
        Skip to Dashboard &rarr;
      </button>

      <div className="relative z-10 w-full max-w-[1120px] mx-8">
        <AnimatePresence mode="wait" custom={dir}>
          <motion.div
            key={slide} custom={dir} variants={slideVariants}
            initial="enter" animate="center" exit="exit"
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] as const }}
            className="bg-paper-50 rounded-2xl p-10 md:p-14 shadow-2xl h-[700px] flex flex-col overflow-hidden max-h-[85vh]"
          >
            {slide === 0 && <SlideTitle />}
            {slide === 1 && <SlideBrief />}
            {slide === 2 && <Slide0 />}
            {slide === 3 && <SlidePipeline />}
            {slide === 4 && <SlideDataset />}
            {slide === 5 && <Slide1 />}
            {slide === 6 && <SlideIGAttribution />}
            {slide === 7 && <SlideABSAEval />}
            {slide === 8 && <Slide2 />}
            {slide === 9 && <Slide3 />}
            {slide === 10 && <SlideLSTMDetect />}
            {slide === 11 && <SlideCUSUMArch />}
            {slide === 12 && <Slide5 />}
            {slide === 13 && <SlideExplainPrimitives />}
            {slide === 14 && <Slide6 />}
            {slide === 15 && <SlideAspectFrequency />}
            {slide === 16 && <SlideCaseStudy />}
            {slide === 17 && <SlideBusinessValue />}
            {slide === 18 && <SlideSavingsWaterfall />}
            {slide === 19 && <SlideDetectionDelayCost />}
            {slide === 20 && <SlideSensitivity />}
            {slide === 21 && <SlideDeployment />}
            {slide === 22 && <SlideProductionML />}
            {slide === 23 && <SlideLimitations />}
            {slide === 24 && <SlideConclusions />}
            {slide === 25 && <SlideTeam />}
            {slide === 26 && <SlideReferences />}
            {slide === 27 && <Slide7 onEnter={onDismiss} />}
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        <div className="flex items-center justify-between mt-5 px-2">
          <button onClick={goPrev} disabled={slide === 0}
            className="text-paper-400 hover:text-paper-50 text-sm font-mono disabled:opacity-30 disabled:cursor-not-allowed transition-colors">
            &larr; Back
          </button>
          <div className="flex gap-2">
            {Array.from({ length: TOTAL_SLIDES }).map((_, i) => (
              <button key={i} onClick={() => { setDir(i > slide ? 1 : -1); setSlide(i) }}
                className={`w-2 h-2 rounded-full transition-all duration-300 ${i === slide ? 'bg-paper-50 scale-125' : 'bg-paper-400/40 hover:bg-paper-400/70'}`}
              />
            ))}
          </div>
          {slide < TOTAL_SLIDES - 1 ? (
            <button onClick={goNext} className="text-paper-50 text-sm font-mono hover:text-signal-amber transition-colors">
              Next &rarr;
            </button>
          ) : <div className="w-16" />}
        </div>
      </div>
    </motion.div>
  )
}
