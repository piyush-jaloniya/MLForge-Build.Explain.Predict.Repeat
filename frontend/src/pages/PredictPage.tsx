import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useModelStore, useSessionStore, useUIStore } from '../store'
import { predictBatch, predictSingle } from '../api/client'
import { cn } from '../lib/utils'
import { 
  Play, Bot, Target, FileText, Download, ListTree, Activity, Rocket, Dices
} from 'lucide-react'

export default function PredictPage() {
  const runs = useModelStore((s) => s.runs)
  const activeRunId = useModelStore((s) => s.activeRunId)
  const metricsMap = useModelStore((s) => s.metrics)
  const session = useSessionStore((s) => s.session)
  const notify = useUIStore((s) => s.notify)

  const completedRuns = runs.filter((r) => r.status === 'complete')
  const [runId, setRunId] = useState(activeRunId || completedRuns[0]?.run_id || '')
  const [inputs, setInputs] = useState<Record<string, string>>({})
  const [result, setResult] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [mode, setMode] = useState<'single' | 'batch' | 'random'>('single')
  const [batchFile, setBatchFile] = useState<File | null>(null)
  const [batchDownloadUrl, setBatchDownloadUrl] = useState('')
  const [batchDownloadName, setBatchDownloadName] = useState('')

  const currentMetrics = runId ? metricsMap[runId] : null
  const featureCols = currentMetrics?.feature_cols || session?.feature_cols || []
  const activeModel = completedRuns.find(r => r.run_id === runId)

  useEffect(() => {
    return () => {
      if (batchDownloadUrl) URL.revokeObjectURL(batchDownloadUrl)
    }
  }, [batchDownloadUrl])

  useEffect(() => {
    setResult(null)
    setBatchFile(null)
    if (batchDownloadUrl) {
      URL.revokeObjectURL(batchDownloadUrl)
      setBatchDownloadUrl('')
      setBatchDownloadName('')
    }
  }, [mode, runId])

  const predict = async () => {
    if (!session || !runId) { notify('error', 'Select a trained model'); return }
    const numeric: Record<string, number> = {}
    for (const col of featureCols) {
      numeric[col] = parseFloat(inputs[col] || '0')
    }
    setLoading(true)
    try {
      const res = await predictSingle(session.session_id, runId, numeric)
      setResult(res.data)
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const doBatchPrediction = async (fileToScore: File, prefix: string = 'predictions') => {
    if (!session || !runId) { notify('error', 'Select a trained model'); return }
    setLoading(true)
    try {
      const res = await predictBatch(runId, session.session_id, fileToScore)
      const blob = res.data as Blob
      if (batchDownloadUrl) URL.revokeObjectURL(batchDownloadUrl)
      const url = URL.createObjectURL(blob)
      setBatchDownloadUrl(url)
      setBatchDownloadName(`${prefix}_${runId.slice(0, 8)}.csv`)
      notify('success', 'Batch predictions ready for download')
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Batch prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const predictFromFile = () => {
    if (!batchFile) { notify('error', 'Choose a CSV file to predict'); return }
    doBatchPrediction(batchFile, 'batch')
  }

  const predictRandom = () => {
    if (!featureCols.length) { notify('error', 'No features to predict'); return }
    let csv = featureCols.join(',') + '\n'
    for (let i = 0; i < 10; i++) {
        csv += featureCols.map(() => (Math.random() * 100).toFixed(2)).join(',') + '\n'
    }
    const blob = new Blob([csv], { type: 'text/csv' })
    const file = new File([blob], 'random_test_data.csv', { type: 'text/csv' })
    doBatchPrediction(file, 'random')
  }

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <Play size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Upload a dataset and train a model before making predictions.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  return (
    <div className="p-6 md:p-10 max-w-5xl mx-auto space-y-6">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 bg-surface border border-border p-6 rounded-3xl shadow-sm">
        <div>
          <h1 className="text-3xl font-bold text-foreground tracking-tight flex items-center gap-3">
             <Rocket className="text-brand-500" size={32} />
             Live Predictions
          </h1>
          <p className="text-slate-500 mt-2 font-medium">Deploy your trained models instantly for single queries or batch scoring.</p>
        </div>
        
        {/* Run Selector */}
        <div className="flex flex-col gap-2 min-w-[250px]">
           <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest pl-1">Target Model</label>
           <div className="flex items-center gap-2 px-3 py-1 bg-surface-hover border border-border rounded-xl focus-within:border-brand-500 transition-colors">
              <Bot size={16} className="text-slate-500" />
              <select 
                value={runId} onChange={(e) => setRunId(e.target.value)}
                className="bg-transparent border-none text-sm font-semibold focus:outline-none text-foreground py-2 outline-none appearance-none flex-1"
              >
                {completedRuns.length === 0 ? (
                  <option value="">No models trained</option>
                ) : (
                  completedRuns.map((r) => <option key={r.run_id} value={r.run_id}>{r.model_name.replace(/_/g, ' ')}</option>)
                )}
              </select>
           </div>
        </div>
      </div>
      
      {/* Mode Toggle */}
      <div className="flex gap-2 p-1.5 bg-surface-hover border border-border rounded-2xl w-fit flex-wrap">
        <button
          onClick={() => setMode('single')}
          className={cn(
            "flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all",
            mode === 'single' 
              ? "bg-brand-500 text-white shadow-sm border border-brand-400" 
              : "text-slate-500 hover:text-foreground border border-transparent"
          )}
        >
          <Target size={16} /> Single Prediction
        </button>
        <button
          onClick={() => setMode('batch')}
          className={cn(
            "flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all",
            mode === 'batch' 
              ? "bg-brand-500 text-white shadow-sm border border-brand-400" 
              : "text-slate-500 hover:text-foreground border border-transparent"
          )}
        >
          <ListTree size={16} /> Batch Prediction
        </button>
        <button
          onClick={() => setMode('random')}
          className={cn(
            "flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-bold transition-all",
            mode === 'random' 
              ? "bg-brand-500 text-white shadow-sm border border-brand-400" 
              : "text-slate-500 hover:text-foreground border border-transparent"
          )}
        >
          <Dices size={16} /> Random Sample
        </button>
      </div>

      <AnimatePresence mode="wait">
        {mode === 'single' && (
          <motion.div 
            key="single"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
            className="grid grid-cols-1 md:grid-cols-12 gap-8"
          >
            {/* Input Form */}
            <div className="col-span-1 md:col-span-7 bg-surface border border-border rounded-3xl p-6 shadow-sm">
              <h3 className="font-bold text-foreground text-sm tracking-wide uppercase mb-6 flex items-center gap-2">
                 <ListTree size={18} className="text-brand-500" /> Feature Inputs
              </h3>
              
              {featureCols.length === 0 ? (
                 <div className="py-10 text-center text-slate-400 font-medium">Select a trained model to see required inputs.</div>
              ) : (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-h-[500px] overflow-y-auto custom-scrollbar pr-2">
                    {featureCols.map((col) => (
                      <div key={col} className="space-y-1.5 focus-within:text-brand-500">
                        <label className="block text-xs font-bold text-slate-500 uppercase tracking-wider transition-colors">{col}</label>
                        <input
                          type="number" step="any" value={inputs[col] || ''}
                          placeholder="0"
                          onChange={(e) => setInputs((prev) => ({ ...prev, [col]: e.target.value }))}
                          className="w-full px-4 py-3 bg-surface-hover border border-border focus:border-brand-500 rounded-xl text-sm font-medium focus:outline-none transition-colors focus:ring-2 focus:ring-brand-500/20"
                        />
                      </div>
                    ))}
                  </div>
                  
                  <button 
                    onClick={predict} disabled={loading || !runId} 
                    className={cn(
                      "w-full py-4 rounded-xl font-extrabold flex items-center justify-center gap-2 transition-all shadow-md active:scale-[0.98] text-base",
                      (loading || !runId) 
                        ? "bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed" 
                        : "bg-brand-500 hover:bg-brand-600 text-white shadow-brand-500/25 border border-brand-400"
                    )}
                  >
                    {loading ? 'Computing Output...' : 'Generate Prediction'}
                  </button>
                </div>
              )}
            </div>

            {/* Results Sidebar */}
            <div className="col-span-1 md:col-span-5 flex flex-col gap-6">
              
              <div className="bg-brand-500/5 border border-brand-500/20 rounded-3xl p-6 relative overflow-hidden shadow-inner flex-1 min-h-[300px] flex flex-col">
                <h3 className="font-bold text-brand-600 dark:text-brand-400 text-sm tracking-wide uppercase mb-6 flex items-center gap-2">
                   <Target size={18} /> Prediction Result Output
                </h3>
                
                <div className="flex-1 flex flex-col items-center justify-center">
                  {!result ? (
                     <div className="text-slate-400 text-center space-y-3">
                        <Activity size={48} className="mx-auto opacity-20" />
                        <p className="text-sm font-medium max-w-[200px]">Fill in the features and run inference to see results.</p>
                     </div>
                  ) : (
                    <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} className="w-full space-y-6">
                      
                      <div className="bg-surface rounded-2xl p-6 border-2 border-brand-500/30 text-center shadow-lg shadow-brand-500/5 relative overflow-hidden">
                         <div className="absolute top-0 right-0 p-3 opacity-5 pointer-events-none">
                           <Target size={64} />
                         </div>
                         <div className="text-xs font-bold text-slate-500 uppercase tracking-widest mb-2">Estimated Target</div>
                         <div className="text-4xl font-extrabold text-foreground tracking-tight drop-shadow-sm">
                           {result.prediction_label ?? String(result.prediction)}
                         </div>
                         {result.prediction_label != null && String(result.prediction_label) !== String(result.raw_prediction ?? result.prediction) && (
                            <div className="text-xs font-medium text-slate-400 mt-2">
                              Raw output: {String(result.raw_prediction ?? result.prediction)}
                            </div>
                         )}
                      </div>

                      {result.confidence != null && (
                        <div className="bg-emerald-500/10 border border-emerald-500/20 rounded-2xl p-4 flex justify-between items-center">
                           <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-widest">Model Confidence</span>
                           <span className="text-2xl font-extrabold text-emerald-600 dark:text-emerald-400">
                             {(result.confidence * 100).toFixed(1)}%
                           </span>
                        </div>
                      )}

                      {result.probabilities && (
                        <div className="bg-surface border border-border rounded-xl p-4 space-y-3">
                          <h4 className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-2">Discrete Probabilities</h4>
                          {Object.entries(result.probabilities)
                            .sort(([, a], [, b]) => (b as number) - (a as number))
                            .map(([cls, prob]) => {
                               const isTop = cls === (result.prediction_label ?? Object.entries(result.probabilities).sort(([, a], [, b]) => (b as number) - (a as number))[0]?.[0])
                               return (
                                 <div key={cls} className="space-y-1.5">
                                    <div className="flex justify-between items-center text-xs">
                                      <span className={cn(isTop ? "font-bold text-foreground" : "font-medium text-slate-500")}>{cls}</span>
                                      <span className={cn(isTop ? "font-bold text-brand-500" : "font-medium text-slate-400")}>{((prob as number) * 100).toFixed(1)}%</span>
                                    </div>
                                    <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                       <motion.div 
                                         initial={{ width: 0 }} animate={{ width: `${(prob as number) * 100}%` }} transition={{ duration: 0.5 }}
                                         className={cn("h-full rounded-full", isTop ? "bg-brand-500" : "bg-slate-400")} 
                                       />
                                    </div>
                                 </div>
                               )
                            })}
                        </div>
                      )}
                    </motion.div>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}

        {mode === 'batch' && (
          <motion.div 
            key="batch"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
            className="bg-surface border border-border rounded-3xl p-8 shadow-sm flex flex-col md:flex-row gap-8"
          >
            <div className="flex-1">
              <h3 className="font-bold text-foreground text-sm tracking-wide uppercase mb-6 flex items-center gap-2">
                 <FileText size={18} className="text-brand-500" /> Batch Processing
              </h3>
              
              <div className="bg-surface-hover/50 border border-border border-dashed rounded-2xl p-8 text-center relative group">
                 <div className="w-16 h-16 bg-surface border border-border shadow-sm rounded-full flex items-center justify-center mx-auto mb-4 group-hover:scale-110 group-hover:border-brand-500 transition-all duration-300">
                    <FileText size={24} className="text-brand-500" />
                 </div>
                 <h4 className="text-lg font-bold text-foreground mb-1">Upload CSV Target Data</h4>
                 <p className="text-sm text-slate-500 mb-6 max-w-sm mx-auto">
                    Provide a CSV file containing identical feature columns to the training set. The model will score all rows.
                 </p>
                 
                 <label className="inline-flex items-center gap-2 px-6 py-3 bg-foreground text-background hover:bg-foreground/90 font-bold rounded-xl cursor-pointer transition-colors shadow-sm">
                    Select File
                    <input type="file" accept=".csv,text/csv" onChange={(e) => setBatchFile(e.target.files?.[0] || null)} className="hidden" />
                 </label>
                 
                 {batchFile && (
                   <div className="mt-4 inline-flex items-center gap-2 px-4 py-2 bg-brand-500/10 text-brand-600 dark:text-brand-400 rounded-lg text-sm font-semibold border border-brand-500/20">
                      <Target size={16} /> {batchFile.name}
                   </div>
                 )}
              </div>
            </div>

            <div className="w-px bg-border hidden md:block" />

            <div className="flex-1 flex flex-col justify-center space-y-6">
              <div className="space-y-4">
                 <button 
                   onClick={predictFromFile} disabled={loading || !runId || !batchFile} 
                   className={cn(
                     "w-full py-4 rounded-xl font-extrabold flex items-center justify-center gap-2 transition-all shadow-md active:scale-[0.98] text-base",
                     (!runId || loading || !batchFile) 
                       ? "bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed" 
                       : "bg-brand-500 hover:bg-brand-600 text-white shadow-brand-500/25 border border-brand-400"
                   )}
                 >
                   {loading ? 'Processing Batch Inference...' : 'Run Batch Inference'}
                 </button>
                 
                 <AnimatePresence>
                   {batchDownloadUrl && (
                     <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
                       <a 
                         href={batchDownloadUrl} download={batchDownloadName} 
                         className="w-full flex items-center justify-center gap-2 py-4 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-bold rounded-xl border border-emerald-500/20 transition-colors shadow-inner"
                       >
                         <Download size={18} /> Download Computed CSV
                       </a>
                     </motion.div>
                   )}
                 </AnimatePresence>
              </div>
            </div>
          </motion.div>
        )}

        {mode === 'random' && (
          <motion.div 
            key="random"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }}
            className="bg-surface border border-border rounded-3xl p-8 shadow-sm flex flex-col items-center justify-center md:min-h-[300px] text-center"
          >
            <h3 className="font-bold text-foreground text-sm tracking-wide uppercase mb-4 flex items-center gap-2">
               <Dices size={18} className="text-brand-500" /> Generate Mock Data
            </h3>
            
            <p className="text-slate-500 max-w-md mx-auto mb-8 font-medium">
              Create a synthetic dataset with 10 random rows based on your {featureCols.length} features. Useful to quickly test your model's inference engine.
            </p>

            <div className="space-y-4 w-full max-w-sm">
               <button 
                 onClick={predictRandom} disabled={loading || !runId} 
                 className={cn(
                   "w-full py-4 rounded-xl font-extrabold flex items-center justify-center gap-2 transition-all shadow-md active:scale-[0.98] text-base",
                   (!runId || loading) 
                     ? "bg-slate-200 dark:bg-slate-800 text-slate-400 cursor-not-allowed" 
                     : "bg-brand-500 hover:bg-brand-600 text-white shadow-brand-500/25 border border-brand-400"
                 )}
               >
                 {loading ? 'Running Random Inference...' : 'Generate & Predict 10 Rows'}
               </button>
               
               <AnimatePresence>
                 {batchDownloadUrl && (
                   <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
                     <a 
                       href={batchDownloadUrl} download={batchDownloadName} 
                       className="w-full flex items-center justify-center gap-2 py-4 mt-2 bg-emerald-500/10 hover:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-bold rounded-xl border border-emerald-500/20 transition-colors shadow-inner"
                     >
                       <Download size={18} /> Download Computed CSV
                     </a>
                   </motion.div>
                 )}
               </AnimatePresence>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

    </div>
  )
}
