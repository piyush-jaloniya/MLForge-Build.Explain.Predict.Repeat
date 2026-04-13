import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useModelStore, useUIStore, useSessionStore } from '../store'
import { getMetrics, exportModel } from '../api/client'
import PlotlyChart from '../components/PlotlyChart'
import api from '../api/client'
import { cn } from '../lib/utils'
import { 
  BarChart2, Download, Play, RefreshCw, Target, 
  GitBranch, Activity, LayoutGrid 
} from 'lucide-react'
import { useNavigate, Link } from 'react-router-dom'

export default function MetricsPage() {
  const activeRunId = useModelStore(s => s.activeRunId)
  const runs = useModelStore(s => s.runs)
  const metricsMap = useModelStore(s => s.metrics)
  const setActiveRun = useModelStore(s => s.setActiveRun)
  const setMetrics = useModelStore(s => s.setMetrics)
  const notify = useUIStore(s => s.notify)
  const session = useSessionStore(s => s.session)

  const [loading, setLoading] = useState(false)
  const [cmChart, setCmChart] = useState<any>(null)
  const [rocChart, setRocChart] = useState<any>(null)
  const [fiChart, setFiChart] = useState<any>(null)

  const completedRuns = runs.filter(r => r.status === 'complete')
  const runId = activeRunId || completedRuns[0]?.run_id
  const metrics = runId ? metricsMap[runId] : null
  const sid = session?.session_id

  useEffect(() => {
    if (!runId) return
    if (!metricsMap[runId]) {
      setLoading(true)
      getMetrics(runId)
        .then(r => setMetrics(runId, r.data))
        .catch(() => notify('error', 'Failed to load metrics'))
        .finally(() => setLoading(false))
    }
    setCmChart(null); setRocChart(null); setFiChart(null)
  }, [runId])

  useEffect(() => {
    if (!runId || !metrics) return
    
    setLoading(true)
    Promise.allSettled([
       metrics.confusion_matrix && !cmChart 
         ? api.get(`/api/viz/confusion-matrix/${runId}`).then(r => setCmChart(r.data.chart)) 
         : Promise.resolve(),
       metrics.metrics?.auc_roc != null && !rocChart 
         ? api.get(`/api/viz/roc-curve/${runId}`, sid ? { params: { session_id: sid } } : {}).then(r => setRocChart(r.data.chart)) 
         : Promise.resolve(),
       !fiChart 
         ? api.get(`/api/viz/feature-importance/${runId}`).then(r => setFiChart(r.data.chart)) 
         : Promise.resolve()
    ]).finally(() => setLoading(false))

  }, [runId, !!metrics])

  const handleExport = async () => {
    if (!runId) return
    try {
      const res = await exportModel(runId)
      const url = URL.createObjectURL(new Blob([res.data]))
      const a = document.createElement('a'); a.href = url
      a.download = `model_${runId.slice(0, 8)}.pkl`; a.click()
      URL.revokeObjectURL(url)
      notify('success', 'Model exported successfully')
    } catch { notify('error', 'Export failed') }
  }

  if (!runId) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <BarChart2 size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No evaluation metrics available</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Train at least one model to see its metrics and charts here.</p>
        <Link to="/train" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Go to Training
        </Link>
      </div>
    )
  }

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-6">
      
      {/* Header & Controls */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-6 bg-surface border border-border p-6 rounded-3xl shadow-sm">
        <div>
          <h1 className="text-3xl font-bold text-foreground tracking-tight flex items-center gap-3">
             <BarChart2 className="text-brand-500" />
             Evaluation Dashboard
          </h1>
          {metrics && (
            <p className="text-slate-500 mt-2 font-medium flex items-center gap-2 flex-wrap">
              <span className="font-bold text-foreground px-2 py-0.5 bg-surface-hover rounded-md border border-border">
                {metrics.model_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
              </span>
              <span className="w-1 h-1 bg-slate-400 rounded-full" />
              <span className="uppercase text-[10px] tracking-widest font-bold bg-brand-500/10 text-brand-500 px-2 py-0.5 rounded">
                {metrics.task_type}
              </span>
              <span className="w-1 h-1 bg-slate-400 rounded-full" />
              <span className="flex items-center gap-1.5 px-2 py-0.5 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 rounded text-xs font-bold uppercase">
                <Target size={12} /> {metrics.target_col}
              </span>
            </p>
          )}
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1 bg-surface-hover border border-border rounded-xl">
             <GitBranch size={16} className="text-slate-500" />
             <select 
               value={runId} onChange={e => setActiveRun(e.target.value)}
               className="bg-transparent border-none text-sm font-semibold focus:outline-none text-foreground py-2 outline-none appearance-none min-w-[200px]"
             >
               {completedRuns.map(r => (
                  <option key={r.run_id} value={r.run_id} className="text-sm">
                    {r.model_name.replace(/_/g, ' ')}
                  </option>
               ))}
             </select>
          </div>

          <button 
             onClick={handleExport}
             className="flex items-center gap-2 px-5 py-2.5 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl text-sm font-bold transition-all shadow-sm active:scale-95 border border-emerald-400"
          >
             <Download size={16} /> Export .pkl
          </button>
          
          <Link 
             to="/predict"
             className="flex items-center gap-2 px-5 py-2.5 bg-brand-500 hover:bg-brand-600 text-white rounded-xl text-sm font-bold transition-all shadow-sm active:scale-95 border border-brand-400"
          >
             <Play size={16} fill="currentColor" /> Live Predict
          </Link>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {loading && !metrics ? (
           <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex flex-col items-center justify-center h-64 text-slate-400 gap-4">
             <RefreshCw size={32} className="animate-spin text-brand-500" />
             <span className="font-bold tracking-widest uppercase text-xs">Loading telemetry...</span>
           </motion.div>
        ) : metrics && (
           <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
             
             {/* Core Metrics Grid */}
             <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
               {Object.entries(metrics.metrics)
                 .filter(([, v]) => typeof v === 'number')
                 .slice(0, 8) 
                 .map(([k, v]) => (
                   <div key={k} className="bg-surface border border-border rounded-2xl p-5 flex flex-col justify-center relative overflow-hidden group hover:border-brand-500/50 hover:shadow-md transition-all">
                      <div className="absolute top-0 right-0 p-4 opacity-[0.03] group-hover:opacity-10 transition-opacity">
                         <Activity size={48} className="text-brand-500" />
                      </div>
                      <div className="relative z-10">
                        <div className="text-3xl font-extrabold text-foreground tracking-tight mb-1">
                           {formatMetric(v as number)}
                        </div>
                        <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{k.replace(/_/g, ' ')}</div>
                      </div>
                   </div>
                 ))
               }
             </div>

             {/* Charts Layout */}
             <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {cmChart && (
                  <div className="bg-surface border border-border rounded-3xl p-5 shadow-sm overflow-hidden flex flex-col min-h-[400px]">
                    <h3 className="font-bold text-sm tracking-widest uppercase text-foreground mb-4 flex items-center gap-2">
                       <LayoutGrid size={16} className="text-brand-500" /> Confusion Matrix
                    </h3>
                    <div className="flex-1 w-full bg-surface-hover/30 rounded-xl relative">
                       <PlotlyChart figure={cmChart} height={350} />
                    </div>
                  </div>
                )}
                
                {rocChart && (
                  <div className="bg-surface border border-border rounded-3xl p-5 shadow-sm overflow-hidden flex flex-col min-h-[400px]">
                    <h3 className="font-bold text-sm tracking-widest uppercase text-foreground mb-4 flex items-center gap-2">
                       <Activity size={16} className="text-brand-500" /> ROC Curve
                    </h3>
                    <div className="flex-1 w-full bg-surface-hover/30 rounded-xl relative">
                       <PlotlyChart figure={rocChart} height={350} />
                    </div>
                  </div>
                )}

                {fiChart && (
                   <div className={cn("bg-surface border border-border rounded-3xl p-5 shadow-sm overflow-hidden flex flex-col min-h-[400px]", (rocChart && cmChart) ? "lg:col-span-2" : "lg:col-span-1")}>
                     <h3 className="font-bold text-sm tracking-widest uppercase text-foreground mb-4 flex items-center gap-2">
                        <BarChart2 size={16} className="text-brand-500" /> Feature Importance
                     </h3>
                     <div className="flex-1 w-full bg-surface-hover/30 rounded-xl relative">
                        <PlotlyChart figure={fiChart} height={rocChart && cmChart ? 400 : 350} />
                     </div>
                   </div>
                )}
             </div>

             {/* Additional Insights */}
             <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                
                {metrics.cv_scores?.length > 0 && (
                  <div className="bg-surface border border-border rounded-3xl p-6 shadow-sm">
                    <h3 className="font-bold text-sm tracking-widest uppercase text-foreground mb-6 flex items-center gap-2">
                       <Activity size={16} className="text-emerald-500" /> Cross-Validation Scores
                    </h3>
                    <div className="space-y-4">
                       {metrics.cv_scores.map((score, i) => (
                         <div key={i} className="flex items-center gap-4">
                           <span className="text-xs font-bold text-slate-500 w-14 tracking-wider uppercase">Fold {i + 1}</span>
                           <div className="flex-1 h-3 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                             <motion.div 
                               initial={{ width: 0 }} 
                               animate={{ width: `${score * 100}%` }} 
                               transition={{ duration: 1, delay: i * 0.1 }}
                               className="h-full bg-emerald-500 rounded-full" 
                             />
                           </div>
                           <span className="text-sm font-bold text-foreground w-12 text-right">{(score * 100).toFixed(1)}%</span>
                         </div>
                       ))}
                       
                       {(() => {
                         const mean = metrics.cv_scores.reduce((a, b) => a + b, 0) / metrics.cv_scores.length
                         const std = Math.sqrt(metrics.cv_scores.reduce((a, v) => a + (v - mean) ** 2, 0) / metrics.cv_scores.length)
                         return (
                           <div className="pt-5 mt-5 border-t border-border flex items-center justify-between">
                             <span className="text-xs font-bold tracking-widest uppercase text-slate-500">Validation Variance</span>
                             <span className="text-sm font-extrabold text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-3 py-1.5 rounded-lg border border-emerald-500/20 shadow-inner">
                               Mean: {(mean * 100).toFixed(2)}% ± {(std * 100).toFixed(2)}%
                             </span>
                           </div>
                         )
                       })()}
                    </div>
                  </div>
                )}
             </div>
             
           </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

function formatMetric(v: number) {
  if (v >= 0 && v <= 1) return (v * 100).toFixed(1) + '%'
  if (Math.abs(v) > 1000) return v.toLocaleString(undefined, { maximumFractionDigits: 0 })
  return v.toFixed(4)
}
