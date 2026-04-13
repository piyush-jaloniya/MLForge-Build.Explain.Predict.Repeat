import { useState, useRef, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { UploadCloud, FileSpreadsheet, Database, Table, ArrowRight, Loader2 } from 'lucide-react'
import { uploadFile, loadSample } from '../api/client'
import { useSessionStore, useUIStore } from '../store'
import { cn } from '../lib/utils'

const SAMPLES = [
  { name: 'iris', label: 'Iris Classification', desc: 'Predict flower species', icon: <Table size={18} /> },
  { name: 'titanic', label: 'Titanic Survival', desc: 'Binary classification', icon: <Database size={18} /> },
  { name: 'housing', label: 'Housing Prices', desc: 'Regression task', icon: <FileSpreadsheet size={18} /> },
]

export default function UploadPage() {
  const navigate = useNavigate()
  const [dragging, setDragging] = useState(false)
  const [uploading, setUploading] = useState(false)
  const fileInput = useRef<HTMLInputElement>(null)
  const setSession = useSessionStore(s => s.setSession)
  const setPreview = useSessionStore(s => s.setPreview)
  const notify = useUIStore(s => s.notify)

  const applyUploadResponse = (data: any) => {
    const column_names: string[] =
      Array.isArray(data.column_names) && data.column_names.length > 0
        ? data.column_names
        : Array.isArray(data.columns)
          ? data.columns.map((c: any) => (typeof c === 'string' ? c : c.name))
          : []

    setSession({
      session_id: data.session_id,
      filename: data.filename,
      n_rows: data.n_rows,
      n_cols: data.n_cols,
      column_names,
      feature_cols: [],
      target_col: null,
      task_type: null,
    })

    const previewRows = data.head ?? data.rows ?? []
    setPreview(previewRows)
  }

  const handleUpload = async (file: File) => {
    setUploading(true)
    try {
      const res = await uploadFile(file)
      applyUploadResponse(res.data)
      notify('success', `Loaded "${res.data.filename}" — ${res.data.n_rows} rows × ${res.data.n_cols} cols`)
      navigate('/preprocess')
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Upload failed')
    } finally { setUploading(false) }
  }

  const handleSample = async (name: string) => {
    setUploading(true)
    try {
      const res = await loadSample(name)
      applyUploadResponse(res.data)
      notify('success', `Loaded sample "${name}"`)
      navigate('/preprocess')
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || `Failed to load sample "${name}"`)
    } finally { setUploading(false) }
  }

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) handleUpload(file)
  }, [])

  return (
    <div className="min-h-full flex items-center justify-center p-6 md:p-12 relative">
      
      {/* Background decorations */}
      <div className="absolute top-0 inset-x-0 h-64 bg-gradient-to-b from-brand-500/5 to-transparent pointer-events-none" />
      <div className="absolute -top-40 -right-40 w-96 h-96 bg-brand-500/20 blur-[100px] rounded-full pointer-events-none" />

      <div className="w-full max-w-4xl relative z-10 flex flex-col gap-10">
        
        {/* Header */}
        <div className="text-center space-y-4">
          <motion.div initial={{ y: -20, opacity: 0 }} animate={{ y: 0, opacity: 1 }}>
            <h1 className="text-4xl md:text-5xl font-extrabold text-foreground tracking-tight">
              Bring your data to life
            </h1>
          </motion.div>
          <motion.p 
            initial={{ y: -10, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.1 }}
            className="text-slate-500 text-lg max-w-2xl mx-auto"
          >
            Upload your CSV, Excel, or Parquet files to instantly start profiling, preprocessing, and training state-of-the-art ML models.
          </motion.p>
        </div>

        {/* Dropzone */}
        <motion.div
           initial={{ scale: 0.95, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.2 }}
           onDragOver={e => { e.preventDefault(); setDragging(true) }}
           onDragLeave={() => setDragging(false)}
           onDrop={onDrop}
           onClick={() => !uploading && fileInput.current?.click()}
           className={cn(
             "relative overflow-hidden group cursor-pointer border-2 border-dashed rounded-3xl p-12 md:p-20 text-center transition-all duration-300 backdrop-blur-sm",
             dragging 
               ? "border-brand-500 bg-brand-500/5 scale-[1.02] shadow-2xl shadow-brand-500/20" 
               : "border-border bg-surface hover:border-brand-500/50 hover:bg-surface-hover/50 hover:shadow-xl",
             uploading && "opacity-60 cursor-not-allowed pointer-events-none"
           )}
        >
          {/* Animated glow on hover */}
          <div className="absolute inset-0 bg-gradient-to-tr from-brand-500/0 via-brand-500/0 to-brand-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
          
          <div className="relative z-10 flex flex-col items-center gap-6">
            <motion.div 
              animate={dragging ? { y: [-5, 5, -5] } : {}}
              transition={{ repeat: Infinity, duration: 2 }}
              className={cn(
                "w-24 h-24 rounded-full flex items-center justify-center transition-colors duration-300",
                dragging ? "bg-brand-500 text-white shadow-lg shadow-brand-500/30" : "bg-brand-500/10 text-brand-500 group-hover:bg-brand-500 group-hover:text-white"
              )}
            >
               {uploading ? <Loader2 size={40} className="animate-spin" /> : <UploadCloud size={40} strokeWidth={1.5} />}
            </motion.div>
            
            <div className="space-y-2">
              <h3 className="text-2xl font-bold text-foreground">
                {uploading ? 'Processing File...' : dragging ? 'Drop it here!' : 'Drag & drop to upload'}
              </h3>
              <p className="text-slate-500 text-sm font-medium">
                or <span className="text-brand-500 border-b border-brand-500/30 group-hover:border-brand-500 pb-0.5 transition-colors">browse files</span> (CSV, JSON, XLXS, Parquet)
              </p>
            </div>
          </div>

          <input
            ref={fileInput} type="file" accept=".csv,.xlsx,.json,.parquet" hidden
            onChange={e => { const f = e.target.files?.[0]; if (f) handleUpload(f) }}
          />
        </motion.div>

        {/* Sample Datasets */}
        <motion.div 
           initial={{ y: 20, opacity: 0 }} animate={{ y: 0, opacity: 1 }} transition={{ delay: 0.3 }}
           className="space-y-6"
        >
          <div className="flex items-center gap-4">
             <div className="h-px flex-1 bg-border/60" />
             <span className="text-xs font-semibold text-slate-400 uppercase tracking-widest">Or try a sample dataset</span>
             <div className="h-px flex-1 bg-border/60" />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
             {SAMPLES.map((sample, i) => (
                <motion.button
                  whileHover={{ y: -4 }}
                  whileTap={{ scale: 0.98 }}
                  key={sample.name}
                  onClick={() => handleSample(sample.name)}
                  disabled={uploading}
                  className="flex items-center justify-between p-4 bg-surface border border-border hover:border-brand-500/40 rounded-2xl group transition-colors text-left"
                >
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center text-slate-500 group-hover:bg-brand-500/10 group-hover:text-brand-500 transition-colors">
                       {sample.icon}
                    </div>
                    <div>
                      <div className="font-semibold text-foreground text-sm">{sample.label}</div>
                      <div className="text-xs text-slate-500 mt-0.5">{sample.desc}</div>
                    </div>
                  </div>
                  <ArrowRight size={16} className="text-slate-300 dark:text-slate-600 group-hover:text-brand-500 group-hover:translate-x-1 transition-all" />
                </motion.button>
             ))}
          </div>
        </motion.div>

      </div>
    </div>
  )
}
