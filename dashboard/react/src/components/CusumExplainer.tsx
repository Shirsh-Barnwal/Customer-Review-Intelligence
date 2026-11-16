import { useRef, Fragment } from 'react'
import { motion } from 'framer-motion'
import { useGSAP } from '@gsap/react'
import gsap from 'gsap'

/* ━━━ CUSUM demo data ━━━
   Real pipeline parameters (k = 0.50, h = 0.50).
   First 4 points are baseline noise, then a sustained positive shift
   triggers accumulation → signal at t₉.                              */

const DK = 0.50, DH = 0.50

const RAW = [0.02, -0.08, 0.06, -0.02, 0.55, 0.60, 0.58, 0.65, 0.70, 0.62]

const DEMO = (() => {
  let sp = 0, sm = 0
  return RAW.map((x, t) => {
    sp = Math.max(0, sp + x - DK)
    sm = Math.max(0, sm - x - DK)
    return { t, x, sp: +sp.toFixed(2), sm: +sm.toFixed(2), sig: sp > DH || sm > DH }
  })
})()

/* ━━━ Pipeline steps ━━━ */

const STEPS: { label: string; math: JSX.Element; accent?: 'teal' | 'red' }[] = [
  {
    label: 'Observe',
    math: <><var>x</var><sub>t</sub></>,
  },
  {
    label: 'Deviation',
    math: (
      <>
        <var>z</var><sub>t</sub>
        <span className="hidden sm:inline">
          {' '}= <var>x</var><sub>t</sub> &minus; &mu;<sub>0</sub>
        </span>
      </>
    ),
  },
  { label: 'Stack', math: <>&plusmn;</> },
  {
    label: 'Accumulate',
    math: <>S<sup>+</sup><sub>t</sub>&thinsp;,&thinsp;S<sup>&minus;</sup><sub>t</sub></>,
    accent: 'teal',
  },
  {
    label: 'Signal',
    math: <>S &gt; h</>,
    accent: 'red',
  },
]

/* ━━━ SVG geometry ━━━ */

const W = 520, SH = 156
const P = { t: 16, r: 16, b: 26, l: 34 }
const PW = W - P.l - P.r   // 470
const PH = SH - P.t - P.b  // 114
const YCEIL = 0.80

const xOf = (i: number) => P.l + (i / (DEMO.length - 1)) * PW
const yOf = (v: number) => P.t + PH * (1 - v / YCEIL)

/* Pre-build SVG paths */
const pts = DEMO.map((d, i) => `${xOf(i).toFixed(1)},${yOf(d.sp).toFixed(1)}`)
const lineD = 'M' + pts.join(' L')
const areaD =
  lineD +
  ` L${xOf(DEMO.length - 1).toFixed(1)},${yOf(0).toFixed(1)}` +
  ` L${xOf(0).toFixed(1)},${yOf(0).toFixed(1)} Z`

/* Threshold crossing fraction (for colour split) */
const sigIdx = DEMO.findIndex(d => d.sig)
const frac =
  sigIdx > 0
    ? (DH - DEMO[sigIdx - 1].sp) / (DEMO[sigIdx].sp - DEMO[sigIdx - 1].sp)
    : 0
const crossX = xOf(sigIdx - 1) + frac * (xOf(sigIdx) - xOf(sigIdx - 1))
const crossNorm = (crossX - P.l) / PW

const threshY = yOf(DH)

/* ━━━ Entrance animation (isolated from parent variants) ━━━ */

const ease = [0.16, 1, 0.3, 1] as const
const variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.07 } },
}
const child = {
  hidden: { opacity: 0, y: 18 },
  show: { opacity: 1, y: 0, transition: { duration: 0.55, ease } },
}

/* ━━━ Component ━━━ */

export default function CusumExplainer() {
  const root = useRef<HTMLDivElement>(null)
  const clip = useRef<SVGRectElement>(null)

  useGSAP(
    () => {
      if (!clip.current) return

      const step = PW / (DEMO.length - 1)
      const tl = gsap.timeline({ repeat: -1, repeatDelay: 4, delay: 1.2 })

      /* reset */
      tl.set(clip.current, { attr: { width: 0 } })
      tl.set('.cq-dot', { attr: { r: 0 } })
      tl.set('.cq-badge', { autoAlpha: 0, scale: 0.85, transformOrigin: 'center center' })

      /* progressive reveal */
      DEMO.forEach((d, i) => {
        tl.to(clip.current!, {
          attr: { width: Math.min((i + 0.6) * step, PW) },
          duration: 0.42,
          ease: 'power2.out',
        })
        tl.to(`.cq-dot-${i}`, {
          attr: { r: 3.5 },
          duration: 0.22,
          ease: 'back.out(1.4)',
        }, '>-0.10')

        if (d.sig && (i === 0 || !DEMO[i - 1].sig)) {
          tl.to('.cq-badge', {
            autoAlpha: 1,
            scale: 1,
            duration: 0.35,
            ease: 'back.out(1.4)',
          })
        }
      })

      /* hold, fade, reset */
      tl.to({}, { duration: 2.8 })
      tl.to('.cq-chart', { opacity: 0, duration: 0.5, ease: 'power1.in' })
      tl.set('.cq-chart', { opacity: 1 })
    },
    { scope: root },
  )

  return (
    <div ref={root}>
      <motion.div
        variants={variants}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: '-40px' }}
      >
        {/* ─── Title ─── */}
        <motion.div variants={child} className="mb-5">
          <h2 className="font-display text-[28px] sm:text-[32px] leading-snug mb-2 text-ink-300">
            Accumulate doubt. <em>Signal&nbsp;shift.</em>
          </h2>
          <p className="text-sm text-ink-100 max-w-lg leading-relaxed">
            Two-sided cumulative sum tallies deviations from each
            product&rsquo;s baseline&thinsp;&mdash;&thinsp;when enough
            evidence accumulates, drift is flagged.
          </p>
        </motion.div>

        {/* ─── Pipeline ─── */}
        <motion.div
          variants={child}
          className="flex items-center overflow-x-auto pb-2 mb-8 -mx-1"
        >
          {STEPS.map((s, i) => (
            <Fragment key={s.label}>
              <div
                className={[
                  'flex flex-col items-center min-w-[88px] px-4 py-3 rounded-lg shrink-0',
                  'transition-colors duration-200',
                  s.accent === 'teal' && 'bg-signal-teal-light ring-1 ring-signal-teal/15',
                  s.accent === 'red'  && 'bg-signal-red-light ring-1 ring-signal-red/15',
                  !s.accent           && 'bg-paper-100 ring-1 ring-paper-300',
                ]
                  .filter(Boolean)
                  .join(' ')}
              >
                <span className="font-mono text-[9px] uppercase tracking-[0.14em] text-ink-50 mb-1.5 select-none">
                  {s.label}
                </span>
                <span
                  className={[
                    'font-mono text-[13px] whitespace-nowrap not-italic',
                    s.accent === 'teal' && 'text-signal-teal',
                    s.accent === 'red'  && 'text-signal-red',
                    !s.accent           && 'text-ink-200',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                >
                  {s.math}
                </span>
              </div>

              {i < STEPS.length - 1 && (
                <svg
                  width="28"
                  height="12"
                  viewBox="0 0 28 12"
                  className="shrink-0 mx-0.5 text-paper-400"
                  aria-hidden
                >
                  <line x1="0" y1="6" x2="20" y2="6" stroke="currentColor" strokeWidth="1.5" />
                  <polygon points="20,3 28,6 20,9" fill="currentColor" />
                </svg>
              )}
            </Fragment>
          ))}
        </motion.div>

        {/* ─── Accumulation chart ─── */}
        <motion.div variants={child} className="cq-chart mb-8">
          <p className="font-mono text-[10px] uppercase tracking-[0.15em] text-ink-50 mb-2.5">
            S<sup>+</sup> accumulation &middot; demo trace
          </p>

          <div className="bg-paper-50 border border-paper-300 rounded-xl px-4 pt-3 pb-2">
            <svg
              viewBox={`0 0 ${W} ${SH}`}
              className="w-full"
              role="img"
              aria-label="Animated CUSUM S-plus accumulation"
            >
              <defs>
                <clipPath id="cq-clip">
                  <rect ref={clip} x={P.l} y={0} width={0} height={SH} />
                </clipPath>
                <linearGradient
                  id="cq-fill"
                  gradientUnits="userSpaceOnUse"
                  x1={P.l}
                  x2={P.l + PW}
                  y1={0}
                  y2={0}
                >
                  <stop offset={Math.max(0, crossNorm - 0.018)} stopColor="#2D7D6F" />
                  <stop offset={Math.min(1, crossNorm + 0.018)} stopColor="#D4483B" />
                </linearGradient>
                <linearGradient
                  id="cq-stroke"
                  gradientUnits="userSpaceOnUse"
                  x1={P.l}
                  x2={P.l + PW}
                  y1={0}
                  y2={0}
                >
                  <stop offset={Math.max(0, crossNorm - 0.01)} stopColor="#2D7D6F" />
                  <stop offset={Math.min(1, crossNorm + 0.01)} stopColor="#D4483B" />
                </linearGradient>
              </defs>

              {/* Threshold line */}
              <line
                x1={P.l}
                x2={W - P.r}
                y1={threshY}
                y2={threshY}
                stroke="#9C9C94"
                strokeDasharray="5 4"
                strokeWidth={1}
              />
              <text
                x={P.l - 7}
                y={threshY + 3.5}
                textAnchor="end"
                fill="#9C9C94"
                fontSize={9}
                fontFamily="JetBrains Mono, monospace"
              >
                h
              </text>

              {/* Baseline */}
              <line
                x1={P.l}
                x2={W - P.r}
                y1={yOf(0)}
                y2={yOf(0)}
                stroke="#E8E5DF"
                strokeWidth={1}
              />

              {/* Area + line (clipped for progressive reveal) */}
              <g clipPath="url(#cq-clip)">
                <path d={areaD} fill="url(#cq-fill)" fillOpacity={0.09} />
                <path
                  d={lineD}
                  fill="none"
                  stroke="url(#cq-stroke)"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </g>

              {/* Data dots (animated independently) */}
              {DEMO.map((d, i) => (
                <circle
                  key={i}
                  className={`cq-dot cq-dot-${i}`}
                  cx={xOf(i)}
                  cy={yOf(d.sp)}
                  r={0}
                  fill={d.sig ? '#D4483B' : '#2D7D6F'}
                  stroke="#FDFCFA"
                  strokeWidth={1.5}
                />
              ))}

              {/* DRIFT badge — positioned at threshold crossing */}
              {sigIdx >= 0 && (
                <g className="cq-badge">
                  <rect
                    x={crossX - 24}
                    y={threshY - 19}
                    width={48}
                    height={14}
                    rx={7}
                    fill="#D4483B"
                    fillOpacity={0.92}
                  />
                  <text
                    x={crossX}
                    y={threshY - 9.5}
                    textAnchor="middle"
                    fill="#FDFCFA"
                    fontSize={7.5}
                    fontFamily="JetBrains Mono, monospace"
                    fontWeight={600}
                    letterSpacing="0.06em"
                  >
                    DRIFT
                  </text>
                </g>
              )}

              {/* Time labels */}
              {DEMO.map((_, i) => {
                const n = i + 1
                const sub = String(n)
                  .split('')
                  .map(d => String.fromCharCode(0x2080 + Number(d)))
                  .join('')
                return (
                  <text
                    key={i}
                    x={xOf(i)}
                    y={SH - 5}
                    textAnchor="middle"
                    fill="#9C9C94"
                    fontSize={8}
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {`t${sub}`}
                  </text>
                )
              })}

              {/* Y-axis zero label */}
              <text
                x={P.l - 7}
                y={yOf(0) + 3.5}
                textAnchor="end"
                fill="#9C9C94"
                fontSize={9}
                fontFamily="JetBrains Mono, monospace"
              >
                0
              </text>
            </svg>
          </div>
        </motion.div>

        {/* ─── Formulas ─── */}
        <motion.div variants={child} className="mb-5">
          <div className="flex flex-wrap gap-x-8 gap-y-1.5 font-mono text-[13px] text-ink-200 mb-3">
            <span>
              <span className="text-signal-teal">S</span>
              <sup className="text-signal-teal">+</sup>
              <sub className="text-ink-50 text-[0.7em]">t</sub>
              {' = max(0, '}
              <span className="text-signal-teal">S</span>
              <sup className="text-signal-teal">+</sup>
              <sub className="text-ink-50 text-[0.7em]">t&minus;1</sub>
              {' + ('}
              <span className="text-ink-300">z</span>
              <sub className="text-ink-50 text-[0.7em]">t</sub>
              {' \u2212 k))'}
            </span>
            <span>
              <span className="text-signal-teal">S</span>
              <sup className="text-signal-teal">&minus;</sup>
              <sub className="text-ink-50 text-[0.7em]">t</sub>
              {' = max(0, '}
              <span className="text-signal-teal">S</span>
              <sup className="text-signal-teal">&minus;</sup>
              <sub className="text-ink-50 text-[0.7em]">t&minus;1</sub>
              {' + (\u2212'}
              <span className="text-ink-300">z</span>
              <sub className="text-ink-50 text-[0.7em]">t</sub>
              {' \u2212 k))'}
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-[11px] text-ink-50">
            <span>&mu;<sub>0</sub> = per-product baseline</span>
            <span className="text-ink-50/40">&middot;</span>
            <span>k = 0.50 slack</span>
            <span className="text-ink-50/40">&middot;</span>
            <span>h = 0.50 threshold</span>
            <span className="text-ink-50/40">&middot;</span>
            <span>per product</span>
          </div>
        </motion.div>

        {/* ─── Footer ─── */}
        <motion.div variants={child} className="flex items-center gap-3 pt-1">
          <div className="w-7 h-7 rounded-full bg-paper-200 flex items-center justify-center shrink-0">
            <span className="font-mono text-sm text-ink-100">&theta;</span>
          </div>
          <span className="font-mono text-[11px] text-ink-50 leading-relaxed">
            0 training samples required &middot; purely statistical &middot;
            no learned parameters
          </span>
        </motion.div>
      </motion.div>
    </div>
  )
}
