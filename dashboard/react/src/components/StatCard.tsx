import { useCountUp } from '../hooks/useCountUp'

interface Props {
  value: number
  label: string
  suffix?: string
  color?: string
  duration?: number
}

export default function StatCard({ value, label, suffix, color = 'text-ink-300', duration = 1000 }: Props) {
  const animated = useCountUp(value, duration)
  return (
    <div className="group">
      <div className={`stat-value text-4xl ${color} mb-1`}>
        {animated.toLocaleString()}{suffix ?? ''}
      </div>
      <div className="text-sm text-ink-100">{label}</div>
    </div>
  )
}
