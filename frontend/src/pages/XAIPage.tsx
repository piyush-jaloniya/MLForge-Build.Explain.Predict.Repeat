import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useModelStore, useSessionStore, useUIStore } from '../store'
import PlotlyChart from '../components/PlotlyChart'
import api from '../api/client'
import { cn } from '../lib/utils'
import { 
  Network, Share2, Orbit, Lightbulb, 
  RefreshCw, BarChart2, CheckCircle2, ScatterChart, ShieldQuestion 
} from 'lucide-react'

type XAITab = 'shap-beeswarm' | 'shap-waterfall' | 'permutation' | 'feature-importance'

export default function XAIPage() {
  const runs = useModelStore((s) => s.runs)
  const activeRunId = useModelStore((s) => s.activeRunId)
  const setActiveRun = useModelStore((s) => s.setActiveRun)
  const session = useSessionStore((s) => s.session)
  const notify = useUIStore((s) => s.notify)

  const completedRuns = runs.filter(r => r.status === 'complete')
  const [runId, setRunId] = useState(activeRunId || completedRuns[0]?.run_id || '')
  const [tab, setTab] = useState<XAITab>('shap-beeswarm')
  const [charts, setCharts] = useState<Record<string, any>>({})
  const [data, setData] = useState<Record<string, any>>({})
  const [loading, setLoading] = useState<Record<string, boolean>>({})

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <Network size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Upload a dataset and train models to access explainability features.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  if (completedRuns.length === 0) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <ShieldQuestion size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No trained models available</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Train at least one model to run explainability algorithms on it.</p>
        <a href="/train" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Go To Training
        </a>
      </div>
    )
  }

  const sid = session.session_id

  const loadChart = async (type: XAITab, rid: string) => {
    const key = `${type}-${rid}`
    if (charts[key] || loading[key]) return
    setLoading(l => ({ ...l, [key]: true }))
    try {
      let res: any
      if (type === 'shap-beeswarm') {
        res = await api.get(`/api/xai/charts/${rid}/shap-beeswarm`, { params: { session_id: sid } })
        setCharts(c => ({ ...c, [key]: res.data.chart }))
      } else if (type === 'shap-waterfall') {
        res = await api.get(`/api/xai/charts/${rid}/shap-waterfall`, { params: { session_id: sid } })
        setCharts(c => ({ ...c, [key]: res.data.chart }))
      } else if (type === 'permutation') {
        res = await api.get(`/api/xai/permutation/${rid}`, { params: { session_id: sid, n_repeats: 10 } })
        setData(d => ({ ...d, [key]: res.data }))
      } else if (type === 'feature-importance') {
        res = await api.get(`/api/viz/feature-importance/${rid}`)
        setCharts(c => ({ ...c, [key]: res.data.chart }))
      }
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || `${type} failed to load`)
    } finally {
      setLoading(l => ({ ...l, [key]: false }))
    }
  }

  useEffect(() => { if (runId && tab) loadChart(tab, runId) }, [runId, tab])

  const key = `${tab}-${runId}`
  const chart = charts[key]
  const rawData = data[key]
  const isLoading = loading[key]

  const tabs: { id: XAITab; label: string; icon: any; tip: string }[] = [
    { id: 'shap-beeswarm', label: 'SHAP Beeswarm', icon: ScatterChart, tip: 'Each point is one sample. Color = feature value (red=high, blue=low). X-axis = impact on model output. Excellent for macro-level feature impact.' },
    { id: 'shap-waterfall', label: 'SHAP Waterfall', icon: Share2, tip: 'Shows how each feature pushes the prediction for a single sample up or down from the base rate. Excellent for micro-level debugging.' },
    { id: 'permutation', label: 'Permutation', icon: Orbit, tip: 'Displays how much the score drops when a feature is randomly shuffled. Higher indicates the model relies heavily on it.' },
    { id: 'feature-importance', label: 'Model Native FI', icon: BarChart2, tip: 'Displays model-native feature importance derived during training (e.g. tree impurity or coefficient magnitude).' },
  ]

  const activeTabDetails = tabs.find(t => t.id === tab)

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-6 flex flex-col min-h-full">
      
      {/* Header Area */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 bg-surface border border-border p-6 rounded-3xl shadow-sm relative overflow-hidden">
        
        {/* Decorative Background Glow */}
        <div className="absolute -top-32 -right-32 w-64 h-64 bg-brand-500/10 blur-[60px] rounded-full pointer-events-none" />

        <div className="relative z-10">
          <h1 className="text-3xl font-bold text-foreground tracking-tight flex items-center gap-3">
             <Network className="text-brand-500" size={32} />
             Explainability Engine
          </h1>
          <p className="text-slate-500 mt-2 font-medium">Understand precisely why your model makes specific predictions.</p>
        </div>
        
        <div className="relative z-10 flex flex-col gap-2 min-w-[250px]">
           <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest pl-1">Target Model</label>
           <div className="flex items-center gap-2 px-3 py-1 bg-surface-hover border border-border rounded-xl focus-within:border-brand-500 transition-colors">
              <CheckCircle2 size={16} className="text-brand-500" />
              <select 
                value={runId} onChange={e => { setRunId(e.target.value); setActiveRun(e.target.value) }}
                className="bg-transparent border-none text-sm font-bold focus:outline-none text-foreground py-2 outline-none appearance-none flex-1"
              >
                {completedRuns.map(r => <option key={r.run_id} value={r.run_id}>{r.model_name.replace(/_/g, ' ')}</option>)}
              </select>
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 flex-1">
        
        {/* Navigation Sidebar */}
        <div className="lg:col-span-3 space-y-4">
           <h3 className="text-xs font-bold uppercase tracking-widest text-slate-500 ml-2">Analysis Methods</h3>
           <div className="bg-surface border border-border rounded-3xl p-3 shadow-sm flex flex-col gap-2 relative z-10">
              {tabs.map(t => {
                 const Icon = t.icon
                 const isActive = tab === t.id
                 return (
                   <button 
                     key={t.id} onClick={() => setTab(t.id)} 
                     className={cn(
                        "flex items-center gap-3 px-4 py-3.5 rounded-2xl font-bold text-sm transition-all text-left group",
                        isActive 
                          ? "bg-brand-500 text-white shadow-md shadow-brand-500/20" 
                          : "text-slate-500 hover:text-foreground hover:bg-surface-hover"
                     )}
                   >
                      <Icon size={18} className={cn(isActive ? "text-white" : "text-slate-400 group-hover:text-brand-500")} />
                      <span>{t.label}</span>
                   </button>
                 )
              })}
           </div>

           {/* Active Tab Helper */}
           <motion.div 
             key={tab} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
             className="bg-brand-500/5 border border-brand-500/20 rounded-3xl p-5 shadow-sm"
           >
              <h4 className="flex items-center gap-2 text-brand-600 dark:text-brand-400 font-bold text-xs uppercase tracking-wide mb-2">
                 <Lightbulb size={16} /> How to read this
              </h4>
              <p className="text-xs text-brand-600/70 dark:text-brand-400/80 leading-relaxed font-medium">
                 {activeTabDetails?.tip}
              </p>
           </motion.div>
        </div>

        {/* Content Area */}
        <div className="lg:col-span-9 bg-surface border border-border rounded-3xl p-6 shadow-sm flex flex-col min-h-[500px] relative overflow-hidden">
           
           {/* Header */}
           <div className="flex border-b border-border pb-4 mb-4">
              <h3 className="font-bold text-foreground text-sm tracking-wide uppercase flex items-center gap-2">
                 {activeTabDetails?.icon && <activeTabDetails.icon size={18} className="text-brand-500" />} {activeTabDetails?.label}
              </h3>
           </div>

           <div className="flex-1 relative flex flex-col">
              <AnimatePresence mode="wait">
                 {isLoading ? (
                    <motion.div 
                      key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="absolute inset-0 flex flex-col items-center justify-center bg-surface/50 backdrop-blur-sm z-10 gap-4"
                    >
                       <RefreshCw size={40} className="animate-spin text-brand-500" />
                       <div className="text-sm font-bold uppercase tracking-widest text-foreground">Computing Explanations...</div>
                       <div className="text-xs font-semibold text-slate-500 text-center max-w-xs">XAI algorithms are computationally expensive. Please wait.</div>
                    </motion.div>
                 ) : (chart || rawData) ? (
                    <motion.div 
                      key="content" initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                      className="flex-1 w-full bg-surface-hover/30 rounded-2xl border border-border flex items-center justify-center p-2 relative overflow-hidden"
                    >
                       {chart && <PlotlyChart figure={chart} height={500} />}
                       
                       {rawData && tab === 'permutation' && (
                         <div className="absolute inset-0 overflow-y-auto p-6 md:p-10 custom-scrollbar flex flex-col">
                           <h3 className="font-bold text-lg text-foreground tracking-tight mb-8">Mean Validation Loss Degradation</h3>
                           <div className="space-y-6 max-w-3xl">
                             {Object.entries(rawData.permutation_importance || {}).slice(0, 15).map(([feat, imp]: [string, any], index) => {
                               const intensity = Math.max(0, Math.min(100, imp.mean * 100))
                               const isPositive = imp.mean > 0
                               return (
                                 <motion.div 
                                    initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * 0.05 }}
                                    key={feat} className="relative group"
                                  >
                                   <div className="flex justify-between items-end mb-2 text-sm">
                                     <span className="font-bold text-foreground">{feat}</span>
                                     <span className="font-mono text-xs text-slate-500 bg-surface-hover px-2 py-1 rounded">
                                        {imp.mean.toFixed(4)} <span className="text-slate-400">±{imp.std.toFixed(4)}</span>
                                     </span>
                                   </div>
                                   <div className="flex gap-1 h-3 w-full bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden relative">
                                     <motion.div 
                                       initial={{ width: 0 }} animate={{ width: `${intensity}%` }} transition={{ duration: 1, delay: 0.2 }}
                                       className={cn(
                                         "h-full rounded-full relative z-10",
                                         isPositive ? "bg-brand-500" : "bg-red-500 pointer-events-none opacity-50"
                                       )} 
                                     />
                                   </div>
                                 </motion.div>
                               )
                             })}
                           </div>
                         </div>
                       )}
                    </motion.div>
                 ) : (
                    <motion.div 
                      key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                      className="absolute inset-0 flex flex-col items-center justify-center gap-4 text-slate-400"
                    >
                       <ShieldQuestion size={64} className="opacity-20" />
                       <div className="text-sm font-medium tracking-wide max-w-[200px] text-center">Data formulation failed. Unable to render explanatory charts.</div>
                    </motion.div>
                 )}
              </AnimatePresence>
           </div>
        </div>

      </div>
    </div>
  )
}
