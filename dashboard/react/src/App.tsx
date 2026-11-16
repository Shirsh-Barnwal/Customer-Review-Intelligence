import { useState } from 'react'
import { Routes, Route, NavLink, useLocation } from 'react-router-dom'
import { AnimatePresence, motion } from 'framer-motion'
import Tutorial from './components/Tutorial'
import CommandCenter from './pages/CommandCenter'
import ProductPulse from './pages/ProductPulse'
import RootCause from './pages/RootCause'
import SystemHealth from './pages/SystemHealth'

const NAV = [
  { to: '/',              label: 'Command Center' },
  { to: '/pulse',         label: 'Product Pulse' },
  { to: '/root-cause',    label: 'Root Cause' },
  { to: '/system-health', label: 'System Health' },
]

export default function App() {
  const location = useLocation()
  const [showTutorial, setShowTutorial] = useState(true)

  return (
    <div className="min-h-screen bg-paper-100 text-ink-300 font-body">
      {/* Tutorial overlay */}
      <AnimatePresence>
        {showTutorial && (
          <Tutorial onDismiss={() => setShowTutorial(false)} />
        )}
      </AnimatePresence>

      {/* Top navigation */}
      <nav className="sticky top-0 z-50 bg-paper-100/90 backdrop-blur-sm border-b border-paper-300">
        <div className="max-w-7xl mx-auto px-8 flex items-center h-14 gap-8">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2.5 shrink-0">
            <div className="w-2 h-2 rounded-full bg-signal-red animate-pulse-dot" />
            <span className="font-display text-lg tracking-tight text-ink-300">
              SentinelDrift
            </span>
          </NavLink>

          {/* Page links */}
          <div className="flex items-center gap-1">
            {NAV.map(({ to, label }) => (
              <NavLink
                key={to}
                to={to}
                end={to === '/'}
                className={({ isActive }) =>
                  `px-3 py-1.5 rounded-full text-sm transition-all duration-300 ${
                    isActive
                      ? 'bg-ink-300 text-paper-50 font-medium'
                      : 'text-ink-50 hover:text-ink-200 hover:bg-paper-200'
                  }`
                }
              >
                {label}
              </NavLink>
            ))}
          </div>

          <div className="flex-1" />

          {/* Presentation mode button */}
          <button
            onClick={() => setShowTutorial(true)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-mono text-ink-50 border border-paper-300 hover:border-ink-50 hover:text-ink-200 transition-all"
          >
            <span className="w-1.5 h-1.5 rounded-full bg-signal-red" />
            Presentation
          </button>
        </div>
      </nav>

      {/* Page content */}
      <main className="max-w-7xl mx-auto px-8 py-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] as const }}
          >
            <Routes location={location}>
              <Route path="/"              element={<CommandCenter />} />
              <Route path="/pulse"         element={<ProductPulse />} />
              <Route path="/root-cause"    element={<RootCause />} />
              <Route path="/system-health" element={<SystemHealth />} />
            </Routes>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Footer */}
      <footer className="border-t border-paper-300 py-6 mt-16">
        <div className="max-w-7xl mx-auto px-8 flex items-center justify-between text-[10px] text-ink-50 font-mono">
          <span>IntelliSys Ltd. &middot; WM9B7 AI & Deep Learning</span>
          <span>SentinelDrift v2.0 &middot; Dual-Detector Early Warning System</span>
        </div>
      </footer>
    </div>
  )
}
