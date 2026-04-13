import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useModelStore, useSessionStore, useUIStore } from '../store'
import { compareModels, getMetrics } from '../api/client'
import { cn } from '../lib/utils'
import { Trophy, RefreshCw, Layers, Award, PlayCircle, BarChart2 } from 'lucide-react'
import { useNavigate } from 'react-router-dom'

export default function ComparePage() {
  const navigate = useNavigate()
  const runs = useModelStore((s) => s.runs)
  const metricsMap = useModelStore((s) => s.metrics)
  const setMetrics = useModelStore((s) => s.setMetrics)
  const setActiveRun = useModelStore((s) => s.setActiveRun)
  const session = useSessionStore((s) => s.session)
  const notify = useUIStore((s) => s.notify)
  const [comparison, setComparison] = useState<any>(null)
  const [loading, setLoading] = useState(false)

  const completedRuns = runs.filter((r) => r.status === 'complete')

  const compare = async () => {
    if (!session || completedRuns.length < 1) { notify('error', 'Need at least one completed run'); return }
    setLoading(true)
    try {
      for (const run of completedRuns) {
        if (!metricsMap[run.run_id]) {
          const r = await getMetrics(run.run_id)
          setMetrics(run.run_id, r.data)
        }
      }
      const res = await compareModels(session.session_id, completedRuns.map((r) => r.run_id))
      setComparison(res.data)
      notify('success', 'Leaderboard updated')
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Comparison failed')
    } finally {
      setLoading(false)
    }
  }

  const taskType = completedRuns[0]?.task_type || 'classification'
  const primaryMetric = taskType === 'classification' ? 'accuracy' : 'r2'
  const extraMetrics = taskType === 'classification'
    ? ['f1_weighted', 'precision_weighted', 'recall_weighted', 'auc_roc']
    : ['mae', 'rmse', 'mse']

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-6">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 bg-surface border border-border p-6 rounded-3xl shadow-sm">
        <div className="flex items-center gap-4">
           <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-2xl flex items-center justify-center">
             <Trophy size={32} />
           </div>
           <div>
             <h1 className="text-3xl font-extrabold text-foreground tracking-tight">Model Leaderboard</h1>
             <p className="text-slate-500 mt-1 font-medium flex items-center gap-2">
               <Layers size={14} /> {completedRuns.length} completed run{completedRuns.length !== 1 ? 's' : ''} ready for comparison
             </p>
           </div>
        </div>
        
        <button 
          onClick={compare} disabled={loading || completedRuns.length === 0}
          className={cn(
             "px-6 py-3 rounded-xl font-bold flex items-center gap-2 transition-all shadow-sm",
             (loading || completedRuns.length === 0)
               ? "bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed"
               : "bg-brand-500 hover:bg-brand-600 text-white shadow-brand-500/25 border border-brand-400 active:scale-95"
          )}
        >
          <RefreshCw size={18} className={cn(loading && "animate-spin")} />
          {loading ? 'Evaluating...' : 'Compare All Models'}
        </button>
      </div>

      <AnimatePresence mode="wait">
        {completedRuns.length === 0 ? (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="bg-surface border-2 border-dashed border-border rounded-3xl p-16 flex flex-col items-center text-center mt-8"
          >
            <div className="w-20 h-20 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mb-6 text-slate-400">
               <Trophy size={32} />
            </div>
            <h2 className="text-xl font-bold text-foreground mb-2">No models to compare</h2>
            <p className="text-slate-500 mb-8 max-w-sm">You need to train at least one model before you can view the leaderboard.</p>
            <button 
               onClick={() => navigate('/train')}
               className="px-6 py-2 bg-brand-500/10 text-brand-600 dark:text-brand-400 hover:bg-brand-500/20 rounded-xl font-bold transition-colors"
            >
               Go to Training
            </button>
          </motion.div>
        ) : (
          <motion.div 
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            className="bg-surface border border-border rounded-3xl overflow-hidden shadow-sm mt-6 flex flex-col"
          >
            <div className="overflow-x-auto custom-scrollbar">
              <table className="w-full text-left whitespace-nowrap">
                <thead className="bg-surface-hover/30 border-b border-border">
                  <tr>
                    <th className="px-6 py-5 font-bold text-slate-500 uppercase tracking-wider text-[10px] w-16">Rank</th>
                    <th className="px-6 py-5 font-bold text-slate-500 uppercase tracking-wider text-[10px]">Model Architecture</th>
                    <th className="px-6 py-5 font-bold text-brand-500 uppercase tracking-wider text-[10px] bg-brand-500/5">{primaryMetric}</th>
                    {extraMetrics.map((m) => (
                       <th key={m} className="px-6 py-5 font-bold text-slate-500 uppercase tracking-wider text-[10px]">
                         {m.replace(/_/g, ' ')}
                       </th>
                    ))}
                    <th className="px-6 py-5 font-bold text-slate-500 uppercase tracking-wider text-[10px]">Duration</th>
                    <th className="px-6 py-5 font-bold text-slate-500 uppercase tracking-wider text-[10px] text-right">Details</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50 bg-background/30">
                  {/* Stable Sort by primary metric */}
                  {[...completedRuns]
                    .sort((a, b) => {
                       const aVal = a.primary_metric ?? 0
                       const bVal = b.primary_metric ?? 0
                       return bVal - aVal
                    })
                    .map((run, i) => {
                      const m = metricsMap[run.run_id]?.metrics || {}
                      
                      // Identify best if explicit comparison exists, otherwise just rank 1
                      let isBest = false
                      if (comparison?.best_run_id === run.run_id) isBest = true
                      else if (!comparison && i === 0) isBest = true

                      const rankEmoji = isBest ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`

                      return (
                        <tr 
                          key={run.run_id} 
                          className={cn(
                             "transition-colors group",
                             isBest ? "bg-amber-500/5 hover:bg-amber-500/10" : "hover:bg-surface-hover/50"
                          )}
                        >
                          <td className="px-6 py-4">
                            <div className={cn(
                               "w-8 h-8 rounded-full flex items-center justify-center font-bold text-lg",
                               isBest && "bg-amber-500/20 shadow-sm shadow-amber-500/20"
                            )}>
                               {rankEmoji}
                            </div>
                          </td>
                          <td className="px-6 py-4">
                             <div className="flex items-center gap-3">
                               <div className={cn("w-2 h-2 rounded-full", isBest ? "bg-amber-500" : "bg-brand-500")} />
                               <span className={cn("font-bold", isBest ? "text-amber-600 dark:text-amber-400" : "text-foreground")}>
                                 {run.model_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                               </span>
                             </div>
                          </td>
                          <td className="px-6 py-4 font-extrabold text-foreground bg-brand-500/5 border-x border-brand-500/10">
                            {run.primary_metric != null ? (
                               <div className="flex items-center gap-2">
                                 <Award size={16} className={isBest ? "text-amber-500" : "text-brand-500"} />
                                 {(run.primary_metric * 100).toFixed(2)}%
                               </div>
                            ) : '—'}
                          </td>
                          
                          {extraMetrics.map((metric) => {
                             const val = m[metric]
                             let display = '—'
                             if (val != null && typeof val === 'number') {
                               if (val >= 0 && val <= 1) display = (val * 100).toFixed(1) + '%'
                               else display = val.toFixed(4)
                             }
                             return (
                               <td key={metric} className="px-6 py-4 font-medium text-slate-500 text-sm">
                                 {display}
                               </td>
                             )
                          })}
                          
                          <td className="px-6 py-4">
                            <span className="inline-flex items-center gap-1.5 px-2 py-1 rounded-md bg-slate-100 dark:bg-slate-800 text-slate-500 text-xs font-bold uppercase tracking-wider">
                               <PlayCircle size={12} />
                               {run.training_time_s != null ? `${run.training_time_s.toFixed(2)}s` : '—'}
                            </span>
                          </td>
                          
                          <td className="px-6 py-4 text-right">
                            <button 
                              onClick={() => { setActiveRun(run.run_id); navigate('/metrics') }}
                               className="inline-flex items-center gap-2 px-4 py-2 bg-brand-500/10 hover:bg-brand-500/20 text-brand-600 dark:text-brand-400 rounded-lg text-xs font-bold uppercase tracking-wider transition-all border border-brand-500/10 hover:border-brand-500/30 active:scale-95"
                            >
                              <BarChart2 size={14} /> Metrics
                            </button>
                          </td>
                        </tr>
                      )
                    })}
                </tbody>
              </table>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
