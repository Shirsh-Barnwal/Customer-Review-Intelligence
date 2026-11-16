const STATE_STYLES: Record<string, { bg: string; text: string; label: string }> = {
  CRITICAL: { bg: 'bg-red-900/15 border-red-900/30', text: 'text-red-900', label: 'Critical' },
  ALERT:    { bg: 'bg-signal-red/10 border-signal-red/25', text: 'text-signal-red', label: 'Alert' },
  WATCH:    { bg: 'bg-signal-amber/10 border-signal-amber/25', text: 'text-signal-amber', label: 'Watch' },
  NORMAL:   { bg: 'bg-signal-teal/10 border-signal-teal/25', text: 'text-signal-teal', label: 'Normal' },
}

export default function AlertBadge({ state, size = 'sm' }: { state: string; size?: 'sm' | 'md' }) {
  const s = STATE_STYLES[state] ?? STATE_STYLES.NORMAL
  const px = size === 'md' ? 'px-3 py-1 text-xs' : 'px-2 py-0.5 text-[10px]'
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border font-medium font-mono uppercase tracking-wide ${px} ${s.bg} ${s.text}`}>
      {state === 'CRITICAL' && <span className="w-1.5 h-1.5 rounded-full bg-red-900 animate-pulse-dot" />}
      {s.label}
    </span>
  )
}
