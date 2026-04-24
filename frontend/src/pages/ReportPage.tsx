import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useModelStore, useSessionStore, useUIStore } from '../store'
import api from '../api/client'
import DropdownSelect from '../components/DropdownSelect'
import { cn } from '../lib/utils'
import {
  FileText, Download, Wand2, Loader2, GitBranch,
  Sparkles, FileText as ReportIcon
} from 'lucide-react'

export default function ReportPage() {
  const runs = useModelStore(s => s.runs)
  const activeRunId = useModelStore(s => s.activeRunId)
  const session = useSessionStore(s => s.session)
  const notify = useUIStore(s => s.notify)

  const completedRuns = runs.filter(r => r.status === 'complete')
  const [runId, setRunId] = useState(activeRunId || completedRuns[0]?.run_id || '')
  const [report, setReport] = useState('')
  const [modelName, setModelName] = useState('')
  const [loading, setLoading] = useState(false)

  const generateReport = async () => {
    if (!session || !runId) { notify('error', 'Need a session and completed run'); return }
    setLoading(true)
    try {
      const r = await api.get(`/api/ai/report/${runId}`, { params: { session_id: session.session_id } })
      setReport(r.data.report)
      setModelName(r.data.model_name)
      notify('success', 'Report successfully generated')
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Report generation failed — check GEMINI_API_KEY')
    } finally { setLoading(false) }
  }

  const downloadReport = () => {
    const blob = new Blob([report], { type: 'text/markdown' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = `ml_report_${modelName}_${Date.now()}.md`
    a.click(); URL.revokeObjectURL(url)
  }

  // Basic custom markdown renderer for specific elements
  const renderMarkdown = (text: string) => {
    if (!text) return null
    return text.split('\n').map((line, i) => {
      if (line.startsWith('### ')) return <h4 key={i} className="font-bold text-lg mt-5 mb-2 text-foreground tracking-tight">{line.slice(4)}</h4>
      if (line.startsWith('## ')) return <h3 key={i} className="font-extrabold text-xl mt-8 mb-4 text-brand-600 dark:text-brand-400 tracking-tight flex items-center gap-2 border-b border-border pb-2 capitalize">{line.slice(3)}</h3>
      if (line.startsWith('# ')) return <h2 key={i} className="font-black text-3xl mb-6 text-foreground tracking-tight">{line.slice(2)}</h2>
      if (line.startsWith('- ') || line.startsWith('* ')) return <li key={i} className="ml-6 mb-1 text-sm text-foreground/80 list-disc list-outside marker:text-brand-500 pl-1">{line.slice(2)}</li>
      if (line.match(/^\d+\./)) return <li key={i} className="ml-6 mb-1 text-sm text-foreground/80 list-decimal list-outside marker:font-bold pl-1 marker:text-slate-400">{line.replace(/^\d+\.\s*/, '')}</li>
      if (line.startsWith('**') && line.endsWith('**')) return <strong key={i} className="font-bold text-foreground block mb-1">{line.slice(2, -2)}</strong>
      if (line.trim() === '') return <div key={i} className="h-3" />

      // Inline styling matching simple bold patterns inside normal text
      let processedLine = line
      const parts = processedLine.split(/(\*\*.*?\*\*)/g)
      return (
        <p key={i} className="text-sm text-foreground/80 leading-relaxed my-2">
          {parts.map((p, j) => {
            if (p.startsWith('**') && p.endsWith('**')) return <strong key={j} className="font-bold text-foreground">{p.slice(2, -2)}</strong>
            return p
          })}
        </p>
      )
    })
  }

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <FileText size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Please upload a dataset and train models to generate reports.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  return (
    <div className="p-6 md:p-10 max-w-5xl mx-auto space-y-6">

      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 bg-surface border border-border p-6 rounded-3xl shadow-sm">
        <div>
          <h1 className="text-3xl font-bold text-foreground tracking-tight flex items-center gap-3">
            <ReportIcon className="text-brand-500" size={32} />
            AI Experiment Report
          </h1>
          <p className="text-slate-500 mt-2 font-medium flex items-center gap-2">
            <Sparkles size={16} className="text-amber-500" /> Auto-generated Markdown report powered by Gemini AI
          </p>
        </div>

        <div className="flex flex-col items-end gap-3">
          <div className="flex flex-wrap items-center justify-end gap-3">
            <DropdownSelect
              value={runId}
              onChange={setRunId}
              ariaLabel="Select completed model run"
              icon={<GitBranch size={16} />}
              placeholder="No models"
              disabled={completedRuns.length === 0}
              options={completedRuns.map(r => ({ value: r.run_id, label: r.model_name.replace(/_/g, ' ') }))}
              buttonClassName="min-w-[150px]"
            />

            <button
              onClick={generateReport} disabled={loading || !runId}
              className={cn(
                "flex items-center gap-2 px-5 py-2.5 rounded-xl text-sm font-bold transition-all shadow-sm",
                (!runId || loading)
                  ? "bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed border border-border/50"
                  : "bg-brand-500 hover:bg-brand-600 text-white active:scale-95 border border-brand-400"
              )}
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Wand2 size={16} />}
              {loading ? 'Drafting...' : 'Generate Report'}
            </button>
          </div>

          <AnimatePresence>
            {report && (
              <motion.button
                initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }}
                onClick={downloadReport}
                className="flex items-center gap-2 px-5 py-2.5 bg-emerald-500 hover:bg-emerald-600 text-white rounded-xl text-sm font-bold transition-all shadow-sm active:scale-95 border border-emerald-400 self-end"
              >
                <Download size={16} /> Download .md
              </motion.button>
            )}
          </AnimatePresence>
        </div>
      </div>

      <AnimatePresence mode="wait">
        {!report && !loading ? (
          <motion.div
            key="empty"
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="bg-surface border-2 border-dashed border-border rounded-3xl p-16 flex flex-col items-center text-center mt-8"
          >
            <div className="w-20 h-20 bg-slate-100 dark:bg-slate-800 rounded-full flex items-center justify-center mb-6 text-slate-400">
              <FileText size={32} />
            </div>
            <h2 className="text-xl font-bold text-foreground mb-2">Initialize Automated Analysis</h2>
            <p className="text-slate-500 mb-2 max-w-md">Select one of your trained models from the dropdown to automatically generate a comprehensive executive summary.</p>
            <div className="flex flex-wrap items-center justify-center gap-3 mt-6 text-xs font-semibold text-slate-400 uppercase tracking-widest">
              <span>Summary</span><span className="w-1 h-1 bg-slate-200 rounded-full" />
              <span>Dataset Insights</span><span className="w-1 h-1 bg-slate-200 rounded-full" />
              <span>Model Analysis</span><span className="w-1 h-1 bg-slate-200 rounded-full" />
              <span>Recommendations</span>
            </div>
          </motion.div>
        ) : loading ? (
          <motion.div
            key="loading"
            initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.95 }}
            className="bg-brand-500/5 border border-brand-500/20 rounded-3xl p-16 flex flex-col items-center text-center mt-8"
          >
            <div className="w-16 h-16 bg-brand-500/10 rounded-full flex items-center justify-center mb-6 relative">
              <div className="absolute inset-0 border-2 border-brand-500/20 border-t-brand-500 rounded-full animate-spin" />
              <Sparkles className="text-brand-500" size={24} />
            </div>
            <h2 className="text-xl font-bold text-brand-600 dark:text-brand-400 mb-2">Analyzing Experiment Results</h2>
            <p className="text-brand-600/60 dark:text-brand-400/60 text-sm max-w-sm">
              The AI is parsing configuration, evaluation metrics, and feature importance telemetry. This usually takes 5-15 seconds.
            </p>
          </motion.div>
        ) : report ? (
          <motion.div
            key="report"
            initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
            className="bg-surface border border-border shadow-md rounded-3xl p-8 md:p-12"
          >
            <div className="prose prose-sm md:prose-base dark:prose-invert max-w-none prose-h3:text-brand-600 dark:prose-h3:text-brand-400 prose-a:text-brand-500">
              {renderMarkdown(report)}
            </div>
          </motion.div>
        ) : null}
      </AnimatePresence>

    </div>
  )
}
