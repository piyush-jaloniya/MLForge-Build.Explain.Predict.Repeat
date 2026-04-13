import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useModelStore, useSessionStore, useUIStore, type SessionInfo } from './store'
import { getRuns, getSession, getSteps, previewData } from './api/client'
import { cn } from './lib/utils'

import {
  Database, LayoutDashboard, Settings2, Sparkles, AlertCircle, TrendingUp,
  Bot, PanelLeft, Moon, Sun, Table2, GitMerge, FileBarChart, MonitorCheck, X
} from 'lucide-react'

import ErrorBoundary from './components/ErrorBoundary'
import AIAssistantDrawer from './components/AIAssistantDrawer'
import UploadPage from './pages/UploadPage'
import PreprocessPage from './pages/PreprocessPage'
import TrainPage from './pages/TrainPage'
import MetricsPage from './pages/MetricsPage'
import PredictPage from './pages/PredictPage'
import ComparePage from './pages/ComparePage'
import VizPage from './pages/VizPage'
import HyperoptPage from './pages/HyperoptPage'
import XAIPage from './pages/XAIPage'
import ReportPage from './pages/ReportPage'
import DataQualityPage from './pages/DataQualityPage'
import RLAdvisorPage from './pages/RLAdvisorPage'

import './styles/app.css'

const NAV_SECTIONS = [
  {
    label: 'DATA', items: [
      { path: '/', label: 'Upload', icon: Database, end: true },
      { path: '/quality', label: 'Data Quality', icon: AlertCircle },
      { path: '/viz', label: 'Visualize', icon: LayoutDashboard },
      { path: '/preprocess', label: 'Preprocess', icon: Settings2 },
    ]
  },
  {
    label: 'MODELS', items: [
      { path: '/rl-advisor', label: 'RL Advisor', icon: Sparkles },
      { path: '/train', label: 'Train', icon: Bot },
      { path: '/hyperopt', label: 'Hyperopt', icon: GitMerge },
      { path: '/compare', label: 'Compare', icon: Table2 },
    ]
  },
  {
    label: 'ANALYSIS', items: [
      { path: '/metrics', label: 'Metrics', icon: TrendingUp },
      { path: '/xai', label: 'Explain (XAI)', icon: MonitorCheck },
      { path: '/predict', label: 'Predict', icon: FileBarChart },
    ]
  },
  {
    label: 'REPORTS', items: [
      { path: '/report', label: 'Auto Report', icon: Sparkles },
    ]
  },
]

export default function App() {
  const { darkMode, toggleDarkMode, toggleAiDrawer, isAiDrawerOpen, notification, clearNotification, sidebarOpen, setSidebarOpen } = useUIStore()
  const session = useSessionStore(s => s.session)
  const setSession = useSessionStore(s => s.setSession)
  const setPreview = useSessionStore(s => s.setPreview)
  const setSteps = useSessionStore(s => s.setSteps)
  const clearSession = useSessionStore(s => s.clearSession)
  const activeRunId = useModelStore(s => s.activeRunId)
  const setRuns = useModelStore(s => s.setRuns)
  const setActiveRun = useModelStore(s => s.setActiveRun)
  const [bootstrapped, setBootstrapped] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  const readPersistedSession = (): SessionInfo | null => {
    try {
      const raw = localStorage.getItem('mlforge-session')
      if (!raw) return null
      const parsed = JSON.parse(raw)
      const persisted = parsed?.state?.session
      if (persisted && typeof persisted.session_id === 'string') {
        return persisted as SessionInfo
      }
    } catch {
    }
    return null
  }

  // Handle dark mode side-effects on document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
  }, [darkMode])

  useEffect(() => {
    const updateViewport = () => setIsMobile(window.innerWidth < 768)
    updateViewport()
    window.addEventListener('resize', updateViewport)
    return () => window.removeEventListener('resize', updateViewport)
  }, [])

  useEffect(() => {
    if (isMobile && sidebarOpen) {
      setSidebarOpen(false)
    }
  }, [isMobile, setSidebarOpen, sidebarOpen])

  useEffect(() => {
    if (notification) { const t = setTimeout(clearNotification, 4500); return () => clearTimeout(t) }
  }, [notification, clearNotification])

  useEffect(() => {
    let cancelled = false

    const restoreContext = async () => {
      let sessionId = session?.session_id
      if (!sessionId) {
        const persisted = readPersistedSession()
        if (persisted?.session_id) {
          setSession(persisted)
          sessionId = persisted.session_id
        }
      }

      if (!sessionId) {
        if (!cancelled) setBootstrapped(true)
        return
      }

      try {
        const sessionResp = await getSession(sessionId)

        if (cancelled) return

        const hydratedSession: SessionInfo = {
          ...(session || readPersistedSession() || {} as SessionInfo),
          ...sessionResp.data,
          column_names: (session?.column_names || readPersistedSession()?.column_names || []),
          feature_cols: sessionResp.data.feature_cols || session?.feature_cols || [],
          target_col: sessionResp.data.target_col ?? session?.target_col ?? null,
          task_type: sessionResp.data.task_type ?? session?.task_type ?? null,
        }
        setSession(hydratedSession)

        const [previewResult, stepsResult, runsResult] = await Promise.allSettled([
          previewData(sessionId),
          getSteps(sessionId),
          getRuns(sessionId),
        ])

        if (cancelled) return

        if (previewResult.status === 'fulfilled') {
          const previewDataRows = previewResult.value.data
          setPreview(previewDataRows.head ?? previewDataRows.rows ?? [])
          if (Array.isArray(previewDataRows.column_names) && previewDataRows.column_names.length > 0) {
            setSession({ ...hydratedSession, column_names: previewDataRows.column_names })
          }
        }

        if (stepsResult.status === 'fulfilled') {
          setSteps(stepsResult.value.data.steps ?? [])
        }

        const runs = runsResult.status === 'fulfilled' ? (runsResult.value.data.runs ?? []) : []
        setRuns(runs)
        const hasPersistedRun = activeRunId && runs.some((run: { run_id: string }) => run.run_id === activeRunId)
        setActiveRun(hasPersistedRun ? activeRunId : runs[0]?.run_id ?? null)
      } catch {
        if (cancelled) return
        clearSession()
        setRuns([])
        setActiveRun(null)
      } finally {
        if (!cancelled) setBootstrapped(true)
      }
    }

    restoreContext()

    return () => {
      cancelled = true
    }
  }, [clearSession, session?.session_id, setActiveRun, setPreview, setRuns, setSession, setSteps, activeRunId])

  if (!bootstrapped) {
    return (
      <div className="min-h-screen bg-background flex flex-col items-center justify-center gap-4 text-foreground">
        <div className="w-12 h-12 border-4 border-brand-500 border-t-transparent rounded-full animate-spin" />
        <span className="text-sm font-medium tracking-wider text-slate-500">RESTORING MLFORGE CONTEXT</span>
      </div>
    )
  }

  return (
    <BrowserRouter>
      <div className="flex h-screen w-full bg-background text-foreground overflow-hidden font-sans antialiased">

        {/* Sidebar */}
        <motion.aside
          initial={false}
          animate={isMobile ? { x: sidebarOpen ? 0 : '-100%' } : { width: sidebarOpen ? 240 : 70 }}
          transition={{ type: 'spring', damping: 28, stiffness: 260 }}
          className={cn(
            "bg-surface border-r border-border flex flex-col shadow-xl shadow-black/5",
            isMobile
              ? "fixed inset-y-0 left-0 z-40 w-[280px] max-w-[82vw]"
              : "relative z-20 flex-shrink-0"
          )}
        >
          <div className="h-20 flex items-center px-3 border-b border-border justify-between whitespace-nowrap overflow-hidden">
            <div className="flex items-center gap-2 min-w-0">
              <img src="/icon.ico" alt="MLForge icon" className="w-12 h-12 rounded-md shrink-0 object-cover" />
              <AnimatePresence>
                {sidebarOpen && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="flex flex-col items-start leading-tight min-w-0"
                  >
                    <span className="font-bold tracking-tight text-foreground text-[18px] truncate">MLForge</span>
                    <span className="text-[9px] font-semibold tracking-[0.12em] text-slate-500 leading-none">Build. Explain. Predict. Repeat</span>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto overflow-x-hidden py-4 custom-scrollbar">
            {NAV_SECTIONS.map(section => (
              <div key={section.label} className="mb-6 px-3">
                {sidebarOpen ? (
                  <div className="px-3 mb-2 text-[10px] font-bold text-slate-400 uppercase tracking-widest">{section.label}</div>
                ) : (
                  <div className="h-4 border-b border-border/50 mb-2 mx-2" />
                )}
                <div className="flex flex-col gap-1">
                  {section.items.map(({ path, label, end, icon: Icon }) => (
                    <NavLink
                      key={path}
                      to={path}
                      end={!!end}
                      className={({ isActive }) => cn(
                        "flex items-center gap-3 px-3 py-2 rounded-lg transition-all whitespace-nowrap group",
                        isActive
                          ? "bg-brand-500/10 text-brand-500 font-semibold"
                          : "text-slate-500 hover:text-foreground hover:bg-surface-hover"
                      )}
                      title={!sidebarOpen ? label : undefined}
                    >
                      <Icon size={18} className={cn("shrink-0 transition-colors cursor-pointer")} />
                      <AnimatePresence>
                        {sidebarOpen && (
                          <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="text-sm">
                            {label}
                          </motion.span>
                        )}
                      </AnimatePresence>
                    </NavLink>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </motion.aside>

        {isMobile && sidebarOpen && (
          <button
            type="button"
            aria-label="Close sidebar backdrop"
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 z-30 bg-slate-950/30 backdrop-blur-[1px] md:hidden"
          />
        )}

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col min-w-0 bg-background relative overflow-hidden">

          {/* Top Bar */}
          <header className="h-16 shrink-0 border-b border-border bg-surface/50 backdrop-blur-md px-3 md:px-6 flex items-center justify-between z-10">
            <div className="flex items-center gap-3 md:gap-4 min-w-0">
              <button
                onClick={() => setSidebarOpen(!sidebarOpen)}
                className="p-2 -ml-2 rounded-md text-slate-500 hover:text-foreground hover:bg-surface-hover transition-colors"
                title="Toggle Sidebar"
              >
                <PanelLeft size={20} />
              </button>

              {session && (
                <div className="hidden sm:flex items-center gap-4 text-xs font-medium bg-surface px-3 py-1.5 rounded-full border border-border shadow-sm min-w-0">
                  <div className="flex items-center gap-2">
                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                    <span className="text-foreground max-w-[200px] truncate">{session.filename}</span>
                  </div>
                  <div className="w-px h-3 bg-border" />
                  <span className="text-slate-500">{session.n_rows.toLocaleString()} rows × {session.n_cols} cols</span>
                </div>
              )}
            </div>

            <div className="flex items-center gap-2 sm:gap-3">
              <button
                onClick={toggleDarkMode}
                className="p-2 rounded-lg text-slate-500 hover:text-foreground bg-surface border border-border hover:border-brand-500/50 shadow-sm transition-all"
                title="Toggle Theme"
              >
                {darkMode ? <Sun size={18} /> : <Moon size={18} />}
              </button>

              <button
                onClick={toggleAiDrawer}
                className={cn(
                  "flex items-center gap-2 px-3 py-2 rounded-lg border shadow-sm transition-all text-sm font-semibold whitespace-nowrap",
                  isAiDrawerOpen
                    ? "bg-brand-500 text-white border-brand-500"
                    : "bg-surface text-brand-500 hover:text-brand-600 border-brand-500/30 hover:border-brand-500 group"
                )}
              >
                <Bot size={18} className={cn(!isAiDrawerOpen && "group-hover:animate-pulse")} />
                <span className="hidden sm:inline">AI Assistant</span>
              </button>
            </div>
          </header>

          {/* Page Routing */}
          <main className="flex-1 overflow-auto relative custom-scrollbar">
            <div className="hidden">
              {/* Preload Plotly dependency styles via PlotlyChart (ensure they load before usage) */}
            </div>
            <ErrorBoundary>
              <Routes>
                <Route path="/" element={<UploadPage />} />
                <Route path="/quality" element={<DataQualityPage />} />
                <Route path="/viz" element={<VizPage />} />
                <Route path="/preprocess" element={<PreprocessPage />} />
                <Route path="/rl-advisor" element={<RLAdvisorPage />} />
                <Route path="/train" element={<TrainPage />} />
                <Route path="/hyperopt" element={<HyperoptPage />} />
                <Route path="/compare" element={<ComparePage />} />
                <Route path="/metrics" element={<MetricsPage />} />
                <Route path="/xai" element={<XAIPage />} />
                <Route path="/predict" element={<PredictPage />} />
                <Route path="/report" element={<ReportPage />} />
              </Routes>
            </ErrorBoundary>
          </main>

        </div>

      </div>

      {/* Global AI Assistant Drawer */}
      <AIAssistantDrawer />

      {/* Global Notifications */}
      <AnimatePresence>
        {notification && (
          <motion.div
            initial={{ opacity: 0, y: 50, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className={cn(
              "fixed bottom-6 right-6 px-4 py-3 rounded-xl shadow-xl border flex items-center gap-3 z-50 text-sm font-medium",
              notification.type === 'success' ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400" :
                notification.type === 'error' ? "bg-red-500/10 border-red-500/20 text-red-600 dark:text-red-400" :
                  "bg-brand-500/10 border-brand-500/20 text-brand-600 dark:text-brand-400"
            )}
          >
            <div className={cn(
              "p-1 rounded-full",
              notification.type === 'success' ? "bg-emerald-500 text-white" :
                notification.type === 'error' ? "bg-red-500 text-white" :
                  "bg-brand-500 text-white"
            )}>
              {notification.type === 'error' ? <AlertCircle size={14} /> : <Bot size={14} />}
            </div>
            <span>{notification.message}</span>
            <button onClick={clearNotification} className="opacity-50 hover:opacity-100 ml-2">
              <X size={14} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>

    </BrowserRouter>
  )
}
