import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSessionStore, useUIStore } from '../store'
import api from '../api/client'
import { cn } from '../lib/utils'
import { ShieldCheck, Target, RefreshCw, AlertCircle, LayoutList, Fingerprint, FileText } from 'lucide-react'

// (Types omitted for brevity, matching previous implementation)
interface ColQuality {
  name: string
  dtype: string
  col_type: string
  null_count: number
  null_pct: number
  unique_count: number
  is_numeric: boolean
  quality_score: number
  issues: string[]
}

interface QualityReport {
  filename: string
  n_rows: number
  n_cols: number
  overall_score: number
  completeness: number
  uniqueness: number
  consistency: number
  columns: ColQuality[]
  duplicate_rows: number
  duplicate_pct: number
  issues_summary: string[]
  recommendations: string[]
}

export default function DataQualityPage() {
  const session = useSessionStore(s => s.session)
  const notify = useUIStore(s => s.notify)
  const [report, setReport] = useState<QualityReport | null>(null)
  const [loading, setLoading] = useState(false)
  const [tab, setTab] = useState<'overview' | 'columns' | 'issues'>('overview')
  const [sortCol, setSortCol] = useState<'name' | 'quality_score' | 'null_pct'>('quality_score')

  useEffect(() => {
    if (session?.session_id) loadReport()
  }, [session?.session_id])

  const loadReport = async () => {
    if (!session) return
    setLoading(true)
    try {
      const r = await api.get(`/api/data/quality/${session.session_id}`)
      setReport(r.data)
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Failed to load quality report')
    } finally { setLoading(false) }
  }

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <FileText size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Please upload a dataset to view its quality report.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  const scoreColor = (score: number) => score >= 80 ? 'text-emerald-500' : score >= 60 ? 'text-amber-500' : 'text-red-500'
  const scoreBgBase = (score: number) => score >= 80 ? 'bg-emerald-500' : score >= 60 ? 'bg-amber-500' : 'bg-red-500'

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground tracking-tight flex items-center gap-3">
            <ShieldCheck className="text-brand-500" size={32} />
            Data Quality Report
          </h1>
          <p className="text-slate-500 mt-2 font-medium">
            <span className="text-foreground">{session.filename}</span> — {session.n_rows.toLocaleString()} rows × {session.n_cols} cols
          </p>
        </div>
        <button 
          onClick={loadReport} 
          disabled={loading}
          className="flex items-center justify-center gap-2 px-5 py-2.5 bg-surface border border-border hover:border-brand-500/50 hover:bg-surface-hover rounded-xl text-sm font-semibold text-foreground transition-all disabled:opacity-50"
        >
          <RefreshCw size={16} className={cn(loading && "animate-spin")} />
          {loading ? 'Analyzing...' : 'Refresh'}
        </button>
      </div>

      {loading && !report && (
        <div className="flex flex-col items-center justify-center py-20 text-slate-500 gap-4">
          <RefreshCw size={40} className="animate-spin text-brand-500" />
          <p className="text-lg font-medium tracking-tight">Profiling your dataset...</p>
        </div>
      )}

      {report && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-8">
          
          {/* Top Metric Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Overall Quality', value: report.overall_score, icon: ShieldCheck },
              { label: 'Completeness', value: report.completeness, icon: Target },
              { label: 'Uniqueness', value: report.uniqueness, icon: Fingerprint },
              { label: 'Consistency', value: report.consistency, icon: LayoutList },
            ].map(({ label, value, icon: Icon }) => (
              <div key={label} className="bg-surface border border-border rounded-2xl p-5 flex flex-col items-center text-center shadow-sm relative overflow-hidden group">
                <div className={cn("absolute inset-0 opacity-0 group-hover:opacity-5 transition-opacity", scoreBgBase(value))} />
                <div className={cn("p-3 rounded-full mb-3", `bg-${scoreColor(value).split('-')[1]}-500/10`)}>
                  <Icon size={24} className={scoreColor(value)} />
                </div>
                <div className={cn("text-3xl font-extrabold tracking-tight mb-1", scoreColor(value))}>
                  {Math.round(value)}%
                </div>
                <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">{label}</div>
                <div className="w-full h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full mt-4 overflow-hidden relative">
                  <motion.div 
                    initial={{ width: 0 }} animate={{ width: `${value}%` }} transition={{ duration: 1, delay: 0.2 }}
                    className={cn("absolute left-0 top-0 bottom-0 rounded-full", scoreBgBase(value))}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Total Rows', value: report.n_rows.toLocaleString() },
              { label: 'Total Columns', value: report.n_cols },
              { label: 'Duplicate Rows', value: `${report.duplicate_rows} (${report.duplicate_pct.toFixed(1)}%)` },
              { label: 'Issues Found', value: report.issues_summary.length },
            ].map(({ label, value }) => (
              <div key={label} className="bg-surface border border-border rounded-xl p-4 flex flex-col justify-center">
                <div className="text-xl font-bold text-foreground">{value}</div>
                <div className="text-xs font-medium text-slate-500">{label}</div>
              </div>
            ))}
          </div>

          {/* Tabs */}
          <div className="flex items-center gap-2 border-b border-border">
            {(['overview', 'columns', 'issues'] as const).map(t => (
              <button 
                key={t} onClick={() => setTab(t)} 
                className={cn(
                  "px-6 py-3 text-sm font-semibold capitalize border-b-2 transition-colors",
                  tab === t ? "border-brand-500 text-brand-500" : "border-transparent text-slate-500 hover:text-foreground"
                )}
              >
                {t}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <AnimatePresence mode="wait">
            <motion.div 
              key={tab}
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} transition={{ duration: 0.15 }}
            >
              
              {tab === 'overview' && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Recommendations */}
                  <div className="bg-brand-500/5 border border-brand-500/20 rounded-2xl p-6">
                    <h3 className="text-sm font-bold text-brand-600 dark:text-brand-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                       <Sparkles size={16} /> AI Recommendations
                    </h3>
                    {report.recommendations.length === 0 ? (
                      <p className="text-slate-500 text-sm">Data looks remarkably clean!</p>
                    ) : (
                      <ul className="space-y-3">
                        {report.recommendations.map((rec, i) => (
                          <li key={i} className="text-sm text-foreground/80 flex items-start gap-3">
                             <div className="min-w-1.5 h-1.5 rounded-full bg-brand-500 mt-1.5" />
                             <span>{rec}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  {/* Issues Summary */}
                  <div className={cn("border rounded-2xl p-6", report.issues_summary.length > 0 ? "bg-red-500/5 border-red-500/20" : "bg-emerald-500/5 border-emerald-500/20")}>
                    <h3 className={cn("text-sm font-bold uppercase tracking-wider mb-4 flex items-center gap-2", report.issues_summary.length > 0 ? "text-red-500" : "text-emerald-500")}>
                       <AlertCircle size={16} /> {report.issues_summary.length > 0 ? 'Critical Issues' : 'No Global Issues Found'}
                    </h3>
                    {report.issues_summary.length === 0 ? (
                      <p className="text-emerald-600 dark:text-emerald-400 text-sm font-medium">All dataset-level quality checks passed successfully.</p>
                    ) : (
                      <ul className="space-y-3">
                        {report.issues_summary.map((issue, i) => (
                          <li key={i} className="text-sm text-red-600 dark:text-red-400 flex items-start gap-3">
                             <div className="min-w-1.5 h-1.5 rounded-full bg-red-500 mt-1.5" />
                             <span>{issue}</span>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>

                  {/* Column Types */}
                  <div className="md:col-span-2 bg-surface border border-border rounded-2xl p-6">
                    <h3 className="text-sm font-bold text-foreground uppercase tracking-wider mb-5">Column Types</h3>
                    <div className="flex flex-wrap gap-4">
                      {Object.entries(
                        report.columns.reduce((acc: Record<string, number>, col) => {
                          acc[col.col_type] = (acc[col.col_type] || 0) + 1; return acc
                        }, {})
                      ).map(([type, count]) => (
                        <div key={type} className="bg-surface-hover border border-border rounded-xl px-5 py-3 flex flex-col items-center min-w-[100px]">
                          <span className="text-2xl font-bold text-foreground">{count}</span>
                          <span className="text-xs text-slate-500 uppercase tracking-widest">{type}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {tab === 'columns' && (
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-sm">
                    <span className="text-slate-500">Sort by:</span>
                    {(['quality_score', 'null_pct', 'name'] as const).map(s => (
                      <button 
                        key={s} onClick={() => setSortCol(s)} 
                        className={cn("px-3 py-1.5 rounded-lg border text-xs font-semibold uppercase tracking-wider transition-colors", sortCol === s ? "bg-brand-500 border-brand-500 text-white" : "bg-surface border-border text-slate-500 hover:text-foreground")}
                      >
                        {s.replace('_', ' ')}
                      </button>
                    ))}
                  </div>

                  <div className="bg-surface border border-border rounded-2xl overflow-x-auto shadow-sm">
                    <table className="w-full text-left text-sm whitespace-nowrap">
                      <thead>
                        <tr className="bg-surface-hover border-b border-border">
                          {['Column', 'Type', 'Null %', 'Unique', 'Quality', 'Issues'].map(h => (
                            <th key={h} className="px-6 py-4 font-bold text-slate-500 uppercase tracking-wider text-[10px]">{h}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-border/50">
                        {[...report.columns]
                          .sort((a, b) => sortCol === 'name' ? a.name.localeCompare(b.name)
                            : sortCol === 'null_pct' ? b.null_pct - a.null_pct
                              : a.quality_score - b.quality_score)
                          .map(col => (
                            <tr key={col.name} className="hover:bg-surface-hover/50 transition-colors">
                              <td className="px-6 py-4 font-bold text-foreground">{col.name}</td>
                              <td className="px-6 py-4">
                                <span className="px-2 py-1 bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400 rounded-md text-[10px] font-bold uppercase">{col.col_type}</span>
                              </td>
                              <td className={cn("px-6 py-4 font-medium", col.null_pct > 20 ? 'text-red-500' : col.null_pct > 5 ? 'text-amber-500' : 'text-emerald-500')}>
                                {col.null_pct.toFixed(1)}%
                              </td>
                              <td className="px-6 py-4 text-slate-600 dark:text-slate-400">{col.unique_count.toLocaleString()}</td>
                              <td className="px-6 py-4">
                                <div className="flex items-center gap-3">
                                  <div className="w-16 h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                                    <div className={cn("h-full", scoreBgBase(col.quality_score))} style={{ width: `${col.quality_score}%` }} />
                                  </div>
                                  <span className={cn("font-bold text-xs", scoreColor(col.quality_score))}>{Math.round(col.quality_score)}%</span>
                                </div>
                              </td>
                              <td className="px-6 py-4 whitespace-normal min-w-[250px]">
                                {col.issues.length === 0 ? (
                                  <span className="text-emerald-500 font-bold text-xs uppercase">Clear</span>
                                ) : (
                                  <div className="flex flex-wrap gap-1.5">
                                    {col.issues.map((iss, j) => (
                                      <span key={j} className="px-2 py-0.5 bg-red-500/10 text-red-500 rounded text-[10px] font-bold uppercase border border-red-500/20">{iss}</span>
                                    ))}
                                  </div>
                                )}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {tab === 'issues' && (
                <div className="space-y-4">
                  {report.issues_summary.length === 0 ? (
                    <div className="bg-emerald-500/5 border border-emerald-500/20 rounded-2xl p-10 flex flex-col items-center justify-center text-center">
                      <ShieldCheck size={48} className="text-emerald-500 mb-4" />
                      <h3 className="text-xl font-bold text-emerald-600 dark:text-emerald-400 mb-2">Pristine Dataset</h3>
                      <p className="text-emerald-600/70 dark:text-emerald-400/70">We couldn't detect any structural or quality issues.</p>
                    </div>
                  ) : (
                    report.issues_summary.map((issue, i) => (
                      <div key={i} className="bg-red-500/5 border border-red-500/20 rounded-xl p-4 flex items-center gap-4">
                        <AlertCircle className="text-red-500 shrink-0" size={20} />
                        <span className="text-sm font-medium text-red-600 dark:text-red-400 leading-relaxed">{issue}</span>
                      </div>
                    ))
                  )}

                  {report.recommendations.length > 0 && (
                    <div className="pt-6">
                      <h3 className="text-sm font-bold text-foreground uppercase tracking-wider mb-4">Recommended Actions</h3>
                      <div className="space-y-3">
                        {report.recommendations.map((rec, i) => (
                          <div key={i} className="bg-brand-500/5 border border-brand-500/20 rounded-xl p-4 flex items-center gap-4">
                            <Sparkles className="text-brand-500 shrink-0" size={18} />
                            <span className="text-sm font-medium text-brand-600 dark:text-brand-400 leading-relaxed">{rec}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

            </motion.div>
          </AnimatePresence>

        </motion.div>
      )}
    </div>
  )
}

function Sparkles(props: any) {
  return <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m12 3-1.912 5.813a2 2 0 0 1-1.275 1.275L3 12l5.813 1.912a2 2 0 0 1 1.275 1.275L12 21l1.912-5.813a2 2 0 0 1 1.275-1.275L21 12l-5.813-1.912a2 2 0 0 1-1.275-1.275L12 3Z"/></svg>
}
