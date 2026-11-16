const ASPECT_COLORS: Record<string, string> = {
  audio: '#7C9EB2', battery: '#C47D5C', build: '#8B7FB5',
  camera: '#B5728A', charging: '#8BAA6E', connectivity: '#C4A94D',
  display: '#A89070', performance: '#6B8CA3', software: '#5B86AB',
}

export { ASPECT_COLORS }

export default function AspectPill({ aspect, size = 'sm' }: { aspect: string; size?: 'sm' | 'md' }) {
  const color = ASPECT_COLORS[aspect] ?? '#6B8CA3'
  const px = size === 'md' ? 'px-3 py-1 text-xs' : 'px-2 py-0.5 text-[10px]'
  return (
    <span
      className={`inline-block rounded-full font-medium capitalize text-white ${px}`}
      style={{ backgroundColor: color }}
    >
      {aspect}
    </span>
  )
}
