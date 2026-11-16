/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"Instrument Serif"', 'Georgia', 'serif'],
        body: ['Manrope', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
      colors: {
        paper: {
          50: '#FDFCFA',
          100: '#FAFAF7',
          200: '#F0ECE5',
          300: '#E8E5DF',
          400: '#D4D0C8',
        },
        ink: {
          50: '#9C9C94',
          100: '#6B6B63',
          200: '#3D3D37',
          300: '#1B1B18',
        },
        signal: {
          red: '#D4483B',
          'red-light': '#F8EEEC',
          teal: '#2D7D6F',
          'teal-light': '#EBF5F3',
          amber: '#C4880D',
          'amber-light': '#FDF6E8',
        },
        aspect: {
          audio: '#7C9EB2',
          battery: '#C47D5C',
          build: '#8B7FB5',
          camera: '#B5728A',
          charging: '#8BAA6E',
          connectivity: '#C4A94D',
          display: '#A89070',
          performance: '#6B8CA3',
          software: '#5B86AB',
        },
      },
      animation: {
        'fade-up': 'fadeUp 0.7s cubic-bezier(0.16, 1, 0.3, 1) both',
        'fade-in': 'fadeIn 0.5s ease both',
        'slide-right': 'slideRight 0.6s cubic-bezier(0.16, 1, 0.3, 1) both',
        'count-up': 'fadeUp 0.4s ease both',
        'pulse-dot': 'pulseDot 2s ease-in-out infinite',
      },
      keyframes: {
        fadeUp: {
          '0%': { opacity: '0', transform: 'translateY(24px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideRight: {
          '0%': { opacity: '0', transform: 'translateX(-16px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        pulseDot: {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.4' },
        },
      },
    },
  },
  plugins: [],
}
