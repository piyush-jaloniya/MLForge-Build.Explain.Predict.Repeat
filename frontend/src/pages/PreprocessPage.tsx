import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { applyStep, undoStep, resetData, selectFeaturesBody } from '../api/client'
import { useSessionStore, useUIStore } from '../store'
import { cn } from '../lib/utils'
import { 
  Settings2, ArrowRight, Undo2, RotateCcw, BoxSelect, 
  Target, Layers, SearchX, CheckSquare, ListChecks, Wand2, Plus, ArrowDownToLine
} from 'lucide-react'

const STEP_GROUPS = [
  {
    label: 'Missing Values',
    icon: SearchX,
    steps: [
      { id: 'fill_mean', label: 'Fill Mean' },
      { id: 'fill_median', label: 'Fill Median' },
      { id: 'fill_mode', label: 'Fill Mode' },
      { id: 'drop_nulls', label: 'Drop Nulls' },
    ],
  },
  {
    label: 'Encoding',
    icon: Layers,
    steps: [
      { id: 'encode_label', label: 'Label Encode' },
      { id: 'encode_onehot', label: 'One-Hot Encode' },
    ],
  },
  {
    label: 'Scaling',
    icon: Wand2,
    steps: [
      { id: 'scale_standard', label: 'Standard Scale' },
      { id: 'scale_minmax', label: 'MinMax Scale' },
      { id: 'scale_robust', label: 'Robust Scale' },
    ],
  },
  {
    label: 'Outliers',
    icon: Target,
    steps: [
      { id: 'remove_outliers_iqr', label: 'Remove IQR Outliers' },
      { id: 'remove_outliers_zscore', label: 'Remove Z-Score Outliers' },
    ],
  },
]

export default function PreprocessPage() {
  const navigate = useNavigate()
  const session = useSessionStore((s) => s.session)
  const preview = useSessionStore((s) => s.preview)
  const steps = useSessionStore((s) => s.steps)
  const setSteps = useSessionStore((s) => s.setSteps)
  const updatePreview = useSessionStore((s) => s.updatePreviewFromApply)
  const setFeatureSelection = useSessionStore((s) => s.setFeatureSelection)
  const notify = useUIStore((s) => s.notify)

  const [selectedCols, setSelectedCols] = useState<string[]>([])
  const [targetCol, setTargetCol] = useState('')
  const [taskType, setTaskType] = useState('classification')
  const [loading, setLoading] = useState(false)
  const [stepCount, setStepCount] = useState(0)

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <Settings2 size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Please upload a dataset to start preprocessing.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  const apply = async (stepType: string) => {
    setLoading(true)
    try {
      const res = await applyStep(session.session_id, stepType, { columns: selectedCols })
      updatePreview(res.data)
      setStepCount(res.data.step_index + 1)
      setSteps([...steps, { step_type: stepType, params: { columns: selectedCols } }])
      notify('success', res.data.message)
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Step failed')
    } finally {
      setLoading(false)
    }
  }

  const undo = async () => {
    setLoading(true)
    try {
      const res = await undoStep(session.session_id)
      updatePreview(res.data)
      setStepCount(res.data.steps_remaining)
      notify('info', res.data.message)
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Undo failed')
    } finally {
      setLoading(false)
    }
  }

  const reset = async () => {
    setLoading(true)
    try {
      const res = await resetData(session.session_id)
      updatePreview(res.data)
      setStepCount(0)
      setSteps([])
      notify('info', 'Reset to original dataset')
    } catch (e: any) {
      notify('error', 'Reset failed')
    } finally {
      setLoading(false)
    }
  }

  const confirmFeatureSelection = async () => {
    if (!targetCol) { notify('error', 'Select a target column'); return }
    if (selectedCols.length === 0) { notify('error', 'Select at least one feature column'); return }
    setLoading(true)
    try {
      await selectFeaturesBody(session.session_id, selectedCols, targetCol, taskType)
      setFeatureSelection(selectedCols, targetCol, taskType)
      notify('success', `${selectedCols.length} features → target "${targetCol}"`)
      navigate('/train')
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Feature selection failed')
    } finally {
      setLoading(false)
    }
  }

  const cols = session.column_names || []
  const previewCols = preview.length > 0 ? Object.keys(preview[0]) : cols

  return (
    <div className="p-6 md:p-10 max-w-screen-2xl mx-auto space-y-6">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
             <Settings2 className="text-brand-500" size={32} />
             Preprocess Pipeline
          </h1>
          <p className="text-slate-500 mt-2 font-medium">
             <span className="text-foreground">{session.filename}</span> — Transform your data before training
          </p>
        </div>
        
        <div className="flex gap-2">
           <button 
             onClick={undo} disabled={loading || stepCount === 0} 
             className="flex items-center gap-2 px-4 py-2 bg-amber-500/10 text-amber-600 hover:bg-amber-500/20 disabled:opacity-50 border border-amber-500/20 rounded-xl text-sm font-semibold transition-all"
           >
             <Undo2 size={16} /> Undo Step
           </button>
           <button 
             onClick={reset} disabled={loading} 
             className="flex items-center gap-2 px-4 py-2 bg-red-500/10 text-red-600 hover:bg-red-500/20 disabled:opacity-50 border border-red-500/20 rounded-xl text-sm font-semibold transition-all"
           >
             <RotateCcw size={16} /> Reset All
           </button>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-8">
        
        {/* Left Sidebar: Controls */}
        <div className="xl:col-span-4 space-y-6">
          
          {/* Column Selector */}
          <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm flex flex-col max-h-[350px]">
            <div className="p-4 bg-surface-hover/50 border-b border-border flex items-center justify-between">
               <div className="flex items-center gap-2">
                 <BoxSelect size={18} className="text-brand-500" />
                 <h3 className="font-bold text-foreground text-sm tracking-wide uppercase">Select Columns</h3>
               </div>
               <div className="flex gap-1.5">
                 <button onClick={() => setSelectedCols(cols)} className="text-[10px] bg-brand-500/10 text-brand-600 dark:bg-brand-500/20 dark:text-brand-400 hover:bg-brand-500/20 dark:hover:bg-brand-500/30 px-3 py-1.5 rounded-md font-bold uppercase transition-colors">All</button>
                 <button onClick={() => setSelectedCols([])} className="text-[10px] bg-brand-500/10 text-brand-600 dark:bg-brand-500/20 dark:text-brand-400 hover:bg-brand-500/20 dark:hover:bg-brand-500/30 px-3 py-1.5 rounded-md font-bold uppercase transition-colors">None</button>
               </div>
            </div>
            <div className="p-2 overflow-y-auto custom-scrollbar flex-1">
              {cols.map((col) => (
                <label key={col} className="flex items-center gap-3 p-2 hover:bg-surface-hover rounded-lg cursor-pointer transition-colors group">
                  <div className="relative flex items-center justify-center">
                    <input 
                      type="checkbox" 
                      className="peer appearance-none w-5 h-5 border-2 border-border rounded checked:bg-brand-500 checked:border-brand-500 transition-colors"
                      checked={selectedCols.includes(col)}
                      onChange={(e) => setSelectedCols(e.target.checked ? [...selectedCols, col] : selectedCols.filter(c => c !== col))} 
                    />
                    <CheckSquare size={14} className="absolute text-white opacity-0 peer-checked:opacity-100 pointer-events-none" strokeWidth={3} />
                  </div>
                  <span className="text-sm font-medium text-foreground group-hover:text-brand-500 transition-colors truncate">{col}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Steps */}
          <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm">
             <div className="p-4 bg-surface-hover/50 border-b border-border flex items-center gap-2">
               <ListChecks size={18} className="text-brand-500" />
               <h3 className="font-bold text-foreground text-sm tracking-wide uppercase">Apply Transforms</h3>
             </div>
             <div className="p-4 space-y-6">
                {STEP_GROUPS.map((group) => {
                  const GroupIcon = group.icon
                  return (
                    <div key={group.label} className="space-y-3">
                      <div className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-widest px-1">
                        <GroupIcon size={14} />
                        {group.label}
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {group.steps.map((s) => (
                          <button 
                            key={s.id} disabled={loading} onClick={() => apply(s.id)} 
                            className="p-2.5 bg-surface-hover hover:bg-brand-500/10 border border-border hover:border-brand-500/50 rounded-xl text-left transition-all group relative overflow-hidden disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <span className="text-xs font-semibold text-foreground group-hover:text-brand-600 dark:group-hover:text-brand-400 relative z-10">{s.label}</span>
                            <div className="absolute right-2 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-all transform group-hover:translate-x-0 -translate-x-2">
                              <Plus size={14} className="text-brand-500" />
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )
                })}
             </div>
          </div>

          {/* Target Selection */}
          <div className="bg-brand-500/5 border border-brand-500/20 rounded-2xl p-5 shadow-sm shadow-brand-500/5 relative overflow-hidden">
             
             <div className="absolute top-0 right-0 p-4 opacity-5 pointer-events-none">
               <Target size={100} />
             </div>

             <h3 className="font-bold text-brand-600 dark:text-brand-400 text-sm tracking-wide uppercase mb-4 relative z-10">Configure & Train</h3>
             
             <div className="space-y-4 relative z-10">
                <div className="space-y-1.5">
                  <label className="block text-xs font-bold tracking-wider text-slate-500 uppercase">Target Column</label>
                  <select 
                    value={targetCol} onChange={(e) => setTargetCol(e.target.value)} 
                    className="w-full px-3 py-2.5 bg-surface border border-brand-500/30 rounded-xl text-sm font-medium focus:outline-none focus:border-brand-500 transition-all"
                  >
                    <option value="">-- Required --</option>
                    {cols.map((c) => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                
                <div className="space-y-1.5">
                  <label className="block text-xs font-bold tracking-wider text-slate-500 uppercase">Task Type</label>
                  <select 
                    value={taskType} onChange={(e) => setTaskType(e.target.value)} 
                    className="w-full px-3 py-2.5 bg-surface border border-form rounded-xl text-sm font-medium focus:outline-none focus:border-brand-500 transition-all border-border"
                  >
                    <option value="classification">Classification</option>
                    <option value="regression">Regression</option>
                  </select>
                </div>

                <div className="pt-2">
                  <button 
                    onClick={confirmFeatureSelection} disabled={loading} 
                    className={cn(
                      "w-full py-3.5 px-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all shadow-md active:scale-[0.98]",
                      loading 
                        ? "bg-slate-200 dark:bg-slate-800 text-slate-400" 
                        : "bg-brand-500 hover:bg-brand-600 text-white shadow-brand-500/25 border border-brand-400"
                    )}
                  >
                    Confirm & Start Training
                    <ArrowRight size={18} />
                  </button>
                </div>
             </div>
          </div>

        </div>

        {/* Right Content: Data Preview */}
        <div className="xl:col-span-8 space-y-4 flex flex-col">
          
          {/* Timeline / Overview */}
          <div className="bg-surface border border-border rounded-xl p-4 flex items-center justify-between text-sm shadow-sm">
             <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-slate-100 dark:bg-slate-800 flex items-center justify-center border border-border">
                  <ArrowDownToLine size={14} className="text-slate-500" />
                </div>
                <div>
                   <span className="font-bold text-foreground">Pipeline State</span>
                   <span className="text-slate-500 ml-2">— {stepCount} transformation{stepCount !== 1 && 's'} applied</span>
                </div>
             </div>
             <div className="flex gap-4 font-medium">
               <span className="text-slate-500">Columns: <span className="text-foreground">{previewCols.length}</span></span>
               <span className="text-slate-500">Selected Features: <span className="text-brand-500">{selectedCols.length}</span></span>
             </div>
          </div>

          <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm flex-1 flex flex-col min-h-[500px]">
            <div className="p-4 bg-surface-hover/50 border-b border-border flex justify-between items-center">
              <h3 className="font-bold text-foreground text-sm tracking-wide uppercase">Table Preview</h3>
              <span className="text-xs font-semibold text-slate-400 bg-surface border border-border px-2 py-0.5 rounded-full">Showing Top 20 Rows</span>
            </div>
            
            <div className="overflow-x-auto flex-1 custom-scrollbar">
              <table className="w-full text-left text-sm whitespace-nowrap">
                <thead className="bg-surface sticky top-0 z-10 shadow-sm shadow-border/50">
                  <tr>
                    {previewCols.map((col) => (
                      <th key={col} className="px-6 py-4 font-bold text-slate-500 uppercase tracking-wider text-[10px] border-b border-border">
                        {col}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-border/50">
                  {preview.slice(0, 20).map((row, i) => (
                    <tr key={i} className="hover:bg-surface-hover/50 transition-colors">
                      {previewCols.map((col) => {
                        const val = String(row[col] ?? '')
                        const isNull = val === 'null' || val === ''
                        return (
                          <td key={col} className="px-6 py-3 min-w-[120px] max-w-[250px] overflow-hidden text-ellipsis border-b border-border/50">
                            {isNull ? (
                              <span className="text-slate-400 italic text-xs">null</span>
                            ) : (
                              <span className="text-foreground font-medium text-xs">{val}</span>
                            )}
                          </td>
                        )
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
              {preview.length === 0 && (
                <div className="flex items-center justify-center h-40 text-slate-500 text-sm">
                  No data to display.
                </div>
              )}
            </div>
          </div>
          
        </div>

      </div>
    </div>
  )
}
