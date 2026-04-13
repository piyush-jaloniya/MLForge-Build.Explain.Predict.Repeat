import { useState, useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { getAvailableModels, startTraining, getTrainStatus, getRuns } from '../api/client'
import { useSessionStore, useModelStore, useUIStore, type RunInfo } from '../store'
import { cn } from '../lib/utils'
import { 
  Bot, Rocket, AlertCircle, CheckCircle2, Play, 
  Settings2, Activity, PlayCircle, Loader2, FastForward, ArrowRight
} from 'lucide-react'

interface ModelDef { name: string; display_name: string; label?: string; task_types: string[]; description?: string }

export default function TrainPage() {
  const navigate = useNavigate()
  const session = useSessionStore((s) => s.session)
  const runs = useModelStore((s) => s.runs)
  const setRuns = useModelStore((s) => s.setRuns)
  const addRun = useModelStore((s) => s.addRun)
  const updateRun = useModelStore((s) => s.updateRun)
  const setActiveRun = useModelStore((s) => s.setActiveRun)
  const notify = useUIStore((s) => s.notify)

  const [models, setModels] = useState<ModelDef[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [loading, setLoading] = useState(false)
  const pollingRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})

  useEffect(() => {
    if (!session) return
    getAvailableModels(session.task_type || undefined).then((r) => setModels(r.data.models || []))
    getRuns(session.session_id).then((r) => {
      const rawRuns = r.data.runs || []
      setRuns(rawRuns)
      rawRuns
        .filter((run: RunInfo) => run.status === 'queued' || run.status === 'running')
        .forEach((run: RunInfo) => {
          if (!pollingRef.current[run.run_id]) pollRun(run.run_id)
        })
    })
  }, [session?.session_id])

  useEffect(() => () => { Object.values(pollingRef.current).forEach(clearInterval) }, [])

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <Bot size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Please upload a dataset to start training models.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  if (!session.target_col) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-amber-500/10 text-amber-500 rounded-full flex items-center justify-center mb-4">
          <Settings2 size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">Feature Selection Required</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">You need to select a target column before training.</p>
        <a href="/preprocess" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Go Validation & Target Selection
        </a>
      </div>
    )
  }

  const pollRun = (runId: string) => {
    pollingRef.current[runId] = setInterval(async () => {
      try {
        const r = await getTrainStatus(runId)
        const d = r.data
        updateRun(runId, {
          status: d.status, progress: d.progress, training_time_s: d.training_time_s,
          primary_metric: d.task_type === 'classification' ? d.metrics?.accuracy : d.metrics?.r2
        })
        if (d.status === 'complete' || d.status === 'failed' || d.status === 'cancelled') {
          clearInterval(pollingRef.current[runId])
          delete pollingRef.current[runId]
          if (d.status === 'complete') notify('success', `${d.model_name} training complete!`)
          else if (d.status === 'failed') notify('error', `Training failed: ${d.error_message}`)
        }
      } catch { }
    }, 1500)
  }

  const train = async () => {
    if (!selectedModel) { notify('error', 'Select a model first'); return }
    setLoading(true)
    try {
      const res = await startTraining({
        session_id: session.session_id,
        model_name: selectedModel,
        feature_cols: session.feature_cols,
        target_col: session.target_col!,
        task_type: session.task_type!,
      })
      const run: RunInfo = {
        run_id: res.data.run_id,
        model_name: selectedModel,
        status: 'queued',
        progress: 0,
        task_type: session.task_type!,
        training_time_s: null,
        primary_metric: null,
        started_at: new Date().toISOString(),
      }
      addRun(run)
      pollRun(res.data.run_id)
      notify('info', `Training "${selectedModel}" started`)
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Training failed to start')
    } finally {
      setLoading(false)
    }
  }

  const taskModels = models.filter(m => !session.task_type || m.task_types.includes(session.task_type))

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-8">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-3 tracking-tight">
             <Bot className="text-brand-500" size={32} />
             Model Training
          </h1>
          <p className="text-slate-500 mt-2 font-medium max-w-2xl flex flex-wrap gap-2 items-center">
             <span>{session.feature_cols.length} features</span>
             <ArrowRight size={14} className="text-brand-500" />
             <strong className="text-foreground">{session.target_col}</strong>
             <span className="w-1 h-1 rounded-full bg-slate-400 mx-2" />
             <span className="px-2 py-0.5 bg-slate-200 dark:bg-slate-800 rounded text-xs font-bold uppercase tracking-wider text-slate-500">{session.task_type}</span>
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        
        {/* Left Col: Algorithm Selection */}
        <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm flex flex-col">
          <div className="p-5 bg-surface-hover/30 border-b border-border flex items-center gap-2">
            <Settings2 size={18} className="text-brand-500" />
            <h3 className="font-bold text-foreground text-sm tracking-wide uppercase">Select Algorithm</h3>
          </div>
          
          <div className="p-5 overflow-y-auto max-h-[500px] custom-scrollbar space-y-3 flex-1">
             {taskModels.length === 0 ? (
               <div className="flex flex-col items-center justify-center py-10 text-slate-400">
                  <Loader2 className="animate-spin mb-2" /> Loading algorithms...
               </div>
             ) : (
               taskModels.map((m) => {
                 const isSelected = selectedModel === m.name
                 return (
                   <label 
                     key={m.name} 
                     className={cn(
                       "flex items-center gap-4 p-4 rounded-xl cursor-pointer transition-all border-2",
                       isSelected 
                         ? "border-brand-500 bg-brand-500/5 shadow-sm" 
                         : "border-transparent bg-surface-hover hover:border-border hover:bg-surface-hover/70"
                     )}
                   >
                     <div className="relative flex items-center justify-center">
                        <input 
                          type="radio" name="model" value={m.name} 
                          checked={isSelected} onChange={() => setSelectedModel(m.name)} 
                          className="peer appearance-none w-5 h-5 rounded-full border-2 border-border checked:border-brand-500 transition-colors shrink-0"
                        />
                        <div className="absolute w-2.5 h-2.5 rounded-full bg-brand-500 opacity-0 peer-checked:opacity-100 transition-opacity pointer-events-none" />
                     </div>
                     <div className="flex-1">
                       <div className="font-bold text-foreground">{m.display_name || m.label || m.name}</div>
                       <div className="text-xs font-semibold text-slate-500 mt-0.5 tracking-wider uppercase">{(m.task_types || []).join(' • ')}</div>
                     </div>
                     {isSelected && (
                       <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}>
                          <CheckCircle2 className="text-brand-500 shrink-0" size={20} />
                       </motion.div>
                     )}
                   </label>
                 )
               })
             )}
          </div>
          
          <div className="p-5 border-t border-border bg-surface-hover/50">
            <button 
              onClick={train} disabled={loading || !selectedModel} 
              className={cn(
                "w-full py-3.5 px-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-md active:scale-[0.98]",
                (loading || !selectedModel)
                  ? "bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed" 
                  : "bg-brand-500 hover:bg-brand-600 border border-brand-400 text-white shadow-brand-500/25"
              )}
            >
              {loading ? (
                <><Loader2 size={18} className="animate-spin" /> Initializing...</>
              ) : (
                <><Rocket size={18} /> Spawn Training Job</>
              )}
            </button>
          </div>
        </div>

        {/* Right Col: Runs List */}
        <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm flex flex-col">
          <div className="p-5 bg-surface-hover/30 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity size={18} className="text-brand-500" />
              <h3 className="font-bold text-foreground text-sm tracking-wide uppercase">Training Runs Tracker</h3>
            </div>
            <div className="px-2 py-0.5 bg-brand-500/10 text-brand-500 rounded text-xs font-bold uppercase tracking-wider">
               {runs.length} Runs
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto max-h-[600px] p-5 space-y-4 custom-scrollbar bg-surface-hover/10">
            <AnimatePresence>
              {runs.length === 0 ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center h-48 text-slate-400 text-center gap-3">
                  <FastForward size={32} className="opacity-20" />
                  <p className="text-sm font-medium tracking-wide">No models trained yet.<br/>Select an algorithm and spawn a job.</p>
                </motion.div>
              ) : (
                runs.map((run) => {
                  const isRunning = run.status === 'queued' || run.status === 'running'
                  const isComplete = run.status === 'complete'
                  const isFailed = run.status === 'failed' || run.status === 'cancelled'
                  
                  return (
                    <motion.div 
                      layout
                      initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
                      key={run.run_id} 
                      onClick={() => { if (isComplete) { setActiveRun(run.run_id); navigate('/metrics') } }}
                      className={cn(
                        "bg-surface border rounded-xl p-4 transition-all relative overflow-hidden group shadow-sm",
                        isComplete ? "border-emerald-500/30 hover:border-emerald-500 hover:shadow-md cursor-pointer" : "border-border",
                        isRunning && "border-brand-500/50 shadow-brand-500/10",
                        isFailed && "border-red-500/30 opacity-70"
                      )}
                    >
                      {/* Interactive hover glow */}
                      {isComplete && <div className="absolute inset-0 bg-gradient-to-tr from-emerald-500/0 to-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />}

                      <div className="relative z-10">
                        <div className="flex justify-between items-center mb-3">
                          <div className="flex items-center gap-3">
                            {isRunning && <div className="w-1.5 h-1.5 rounded-full bg-brand-500 animate-pulse" />}
                            {isComplete && <CheckCircle2 size={16} className="text-emerald-500" />}
                            {isFailed && <AlertCircle size={16} className="text-red-500" />}
                            <span className="font-bold text-foreground tracking-tight">{run.model_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}</span>
                          </div>
                          
                          <span className={cn(
                            "text-[10px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-md",
                            isRunning ? "bg-brand-500/10 text-brand-500 animate-pulse" :
                            isComplete ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" :
                            "bg-red-500/10 text-red-600 dark:text-red-400"
                          )}>
                            {run.status}
                          </span>
                        </div>
                        
                        {isRunning && (
                          <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden w-full">
                            <motion.div 
                               className="h-full bg-brand-500" 
                               initial={{ width: 0 }} 
                               animate={{ width: `${run.progress}%` }} 
                               transition={{ type: "tween" }}
                            />
                          </div>
                        )}
                        
                        {isComplete && run.primary_metric != null && (
                          <div className="flex items-center gap-4 text-xs font-semibold text-slate-500 bg-emerald-500/5 px-3 py-2 rounded-lg border border-emerald-500/10">
                            <div>
                              <span className="uppercase text-emerald-500 tracking-wider">
                                {run.task_type === 'classification' ? 'Accuracy' : 'R²'}
                              </span>
                              <span className="text-foreground ml-2 text-sm">{(run.primary_metric * 100).toFixed(1)}%</span>
                            </div>
                            {run.training_time_s && (
                              <div className="border-l border-emerald-500/20 pl-4 text-emerald-600/70 dark:text-emerald-400/70 uppercase tracking-widest text-[10px] flex items-center gap-1">
                                <PlayCircle size={12} /> {run.training_time_s.toFixed(1)}s
                              </div>
                            )}
                          </div>
                        )}
                        
                        {isComplete && (
                          <div className="text-[10px] font-bold uppercase tracking-wider text-emerald-500 mt-3 transition-transform flex items-center gap-1 translate-x-[-10px] group-hover:translate-x-0 duration-300">
                             View Evaluation Metrics <ArrowRight size={12} />
                          </div>
                        )}
                      </div>
                    </motion.div>
                  )
                })
              )}
            </AnimatePresence>
          </div>
        </div>
        
      </div>
    </div>
  )
}
