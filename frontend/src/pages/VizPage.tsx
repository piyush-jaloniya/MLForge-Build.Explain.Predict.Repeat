import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSessionStore, useUIStore } from '../store'
import PlotlyChart from '../components/PlotlyChart'
import api from '../api/client'
import { cn } from '../lib/utils'
import { 
  BarChart3, Settings2, BoxSelect, ScatterChart, 
  TableProperties, SearchX, LineChart, SlidersHorizontal, Image, MousePointerClick 
} from 'lucide-react'

type ChartType = 'histogram' | 'boxplot' | 'scatter' | 'correlation' | 'missing' | 'pairplot'

const CHART_TYPES: { id: ChartType; label: string; icon: any; desc: string }[] = [
  { id: 'histogram', label: 'Distribution', icon: BarChart3, desc: 'View single data variable distribution' },
  { id: 'boxplot', label: 'Box Plot', icon: BoxSelect, desc: 'Identify outliers and quartiles' },
  { id: 'scatter', label: 'Scatter Plot', icon: ScatterChart, desc: 'Analyze relationship between two variables' },
  { id: 'correlation', label: 'Correlation', icon: TableProperties, desc: 'Heatmap of variable relationships' },
  { id: 'pairplot', label: 'Pair Matrix', icon: LineChart, desc: 'Grid of scatter plots for all features' },
  { id: 'missing', label: 'Missing Values', icon: SearchX, desc: 'Visual map of null values' },
]

export default function VizPage() {
  const session = useSessionStore((s) => s.session)
  const notify = useUIStore((s) => s.notify)
  const [activeChart, setActiveChart] = useState<ChartType>('histogram')
  const [charts, setCharts] = useState<Record<string, any>>({})
  const [loading, setLoading] = useState<Record<string, boolean>>({})
  const [col1, setCol1] = useState('')
  const [col2, setCol2] = useState('')
  const [colorCol, setColorCol] = useState('')
  const [bins, setBins] = useState(30)

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <BarChart3 size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Please upload a dataset to visualize its data patterns.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  const cols = session.column_names || []
  const sid = session.session_id

  const fetchChart = async (type: ChartType, extraParams: Record<string, any> = {}) => {
    const key = `${type}-${JSON.stringify(extraParams)}`
    if (charts[key]) return
    setLoading(l => ({ ...l, [key]: true }))
    try {
      const baseParams = { session_id: sid }
      const allParams = {
        ...baseParams,
        ...Object.fromEntries(Object.entries(extraParams).filter(([, v]) => v !== '' && v != null))
      }
      const ENDPOINT_MAP: Record<ChartType, string> = {
        histogram: '/api/viz/histogram',
        boxplot: '/api/viz/boxplot',
        scatter: '/api/viz/scatter',
        correlation: '/api/viz/correlation-heatmap',
        missing: '/api/viz/missing-values',
        pairplot: '/api/viz/pairplot',
      }
      const res = await api.get(ENDPOINT_MAP[type], { params: allParams })
      setCharts(c => ({ ...c, [key]: res.data.chart }))
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Chart generation failed')
    } finally {
      setLoading(l => ({ ...l, [key]: false }))
    }
  }

  const currentKey = (() => {
    if (activeChart === 'histogram') return `histogram-${JSON.stringify({ column: col1, bins })}`
    if (activeChart === 'boxplot') return `boxplot-${JSON.stringify({ column: col1, group_by: col2 })}`
    if (activeChart === 'scatter') return `scatter-${JSON.stringify({ x: col1, y: col2, color: colorCol })}`
    if (activeChart === 'correlation') return `correlation-${JSON.stringify({})}`
    if (activeChart === 'missing') return `missing-${JSON.stringify({})}`
    if (activeChart === 'pairplot') return `pairplot-${JSON.stringify({ color_col: colorCol })}`
    return ''
  })()

  const currentChart = charts[currentKey]
  const currentLoading = loading[currentKey]

  const handleGenerate = () => {
    if (activeChart === 'histogram') fetchChart('histogram', { column: col1 || cols[0], bins })
    else if (activeChart === 'boxplot') fetchChart('boxplot', { column: col1 || cols[0], group_by: col2 })
    else if (activeChart === 'scatter') fetchChart('scatter', { x: col1 || cols[0], y: col2 || cols[1], color: colorCol })
    else if (activeChart === 'correlation') fetchChart('correlation')
    else if (activeChart === 'missing') fetchChart('missing')
    else if (activeChart === 'pairplot') fetchChart('pairplot', { color_col: colorCol })
  }

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-6">
      <div className="mb-4">
        <h1 className="text-3xl font-bold text-foreground flex items-center gap-3">
           <BarChart3 className="text-brand-500" size={32} />
           Exploratory Data Analysis
        </h1>
        <p className="text-slate-500 mt-2 font-medium">
           Generate high-quality visualizations to understand your dataset's underlying structures.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        
        {/* Left Column: Controls */}
        <div className="lg:col-span-4 space-y-6">
          
          <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm">
            <div className="p-4 bg-surface-hover/50 border-b border-border flex items-center gap-2">
               <Settings2 size={18} className="text-brand-500" />
               <h3 className="font-bold text-foreground text-sm tracking-wide uppercase">Chart Type</h3>
            </div>
            <div className="p-2 grid grid-cols-2 gap-2">
              {CHART_TYPES.map(t => {
                const Icon = t.icon
                const isActive = activeChart === t.id
                return (
                  <button 
                    key={t.id} 
                    onClick={() => setActiveChart(t.id)} 
                    className={cn(
                      "p-3 rounded-xl flex flex-col gap-2 items-start text-left transition-all border",
                      isActive 
                        ? "bg-brand-500/10 border-brand-500 text-brand-500 shadow-sm" 
                        : "bg-surface border-transparent text-slate-500 hover:text-foreground hover:bg-surface-hover hover:border-border"
                    )}
                  >
                    <Icon size={20} className={cn(isActive ? "text-brand-500" : "text-slate-400")} />
                    <div>
                      <div className="font-bold text-xs">{t.label}</div>
                    </div>
                  </button>
                )
              })}
            </div>
          </div>

          <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm">
            <div className="p-4 bg-surface-hover/50 border-b border-border flex items-center gap-2">
               <SlidersHorizontal size={18} className="text-brand-500" />
               <h3 className="font-bold text-foreground text-sm tracking-wide uppercase">Configuration</h3>
            </div>
            
            <div className="p-5 space-y-5">
              {(activeChart === 'histogram' || activeChart === 'boxplot' || activeChart === 'scatter') && (
                <div className="space-y-2">
                  <label className="block text-xs font-bold tracking-wider text-slate-500 uppercase">
                    {activeChart === 'scatter' ? 'X-Axis Column' : 'Primary Column'}
                  </label>
                  <select 
                    value={col1} onChange={e => setCol1(e.target.value)} 
                    className="w-full px-3 py-2.5 bg-surface border border-border rounded-lg text-sm font-medium focus:outline-none focus:border-brand-500 focus:ring-1 focus:ring-brand-500/50 transition-all"
                  >
                    <option value="">-- Autoselect --</option>
                    {cols.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              )}

              {activeChart === 'histogram' && (
                <div className="space-y-3 pt-2">
                  <div className="flex justify-between items-center text-xs font-bold tracking-wider text-slate-500 uppercase">
                    <label>Bin Count</label>
                    <span className="text-brand-500 bg-brand-500/10 px-2 py-0.5 rounded">{bins}</span>
                  </div>
                  <input 
                    type="range" min={5} max={100} value={bins} onChange={e => setBins(+e.target.value)}
                    className="w-full accent-brand-500" 
                  />
                </div>
              )}

              {(activeChart === 'boxplot' || activeChart === 'scatter') && (
                <div className="space-y-2 pt-2">
                  <label className="block text-xs font-bold tracking-wider text-slate-500 uppercase flex items-center gap-2">
                    {activeChart === 'scatter' ? 'Y-Axis Column' : 'Group By Column'}
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-400">OPTIONAL</span>
                  </label>
                  <select 
                    value={col2} onChange={e => setCol2(e.target.value)} 
                    className="w-full px-3 py-2.5 bg-surface border border-border rounded-lg text-sm font-medium focus:outline-none focus:border-brand-500 transition-all"
                  >
                    <option value="">-- None --</option>
                    {cols.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              )}

              {(activeChart === 'scatter' || activeChart === 'pairplot') && (
                <div className="space-y-2 pt-2">
                  <label className="block text-xs font-bold tracking-wider text-slate-500 uppercase flex items-center gap-2">
                    Color Grouping
                    <span className="text-[9px] px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 text-slate-400">OPTIONAL</span>
                  </label>
                  <select 
                    value={colorCol} onChange={e => setColorCol(e.target.value)} 
                    className="w-full px-3 py-2.5 bg-surface border border-border rounded-lg text-sm font-medium focus:outline-none focus:border-brand-500 transition-all"
                  >
                    <option value="">-- None --</option>
                    {cols.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
              )}

              {(activeChart === 'correlation' || activeChart === 'missing') && (
                <div className="p-4 rounded-xl bg-surface-hover border border-border/50 text-center text-sm text-slate-500">
                  This chart type computes automatically over the entire dataset. No extra parameters required.
                </div>
              )}

              <button 
                onClick={handleGenerate} 
                disabled={currentLoading} 
                className={cn(
                  "w-full py-3 px-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all mt-6 shadow-sm shadow-brand-500/20 active:scale-[0.98]",
                  currentLoading 
                    ? "bg-slate-100 dark:bg-slate-800 text-slate-400 cursor-not-allowed" 
                    : "bg-brand-500 hover:bg-brand-600 border border-brand-400/50 text-white"
                )}
              >
                {currentLoading ? (
                  <>
                    <div className="w-4 h-4 border-2 border-slate-400 border-t-transparent rounded-full animate-spin" />
                    Generating...
                  </>
                ) : (
                  <>
                    <Image size={18} />
                    Render Chart
                  </>
                )}
              </button>
            </div>
          </div>

        </div>

        {/* Right Column: Visualization Canvas */}
        <div className="lg:col-span-8">
           <div className="bg-surface border border-border rounded-2xl shadow-sm min-h-[600px] flex flex-col overflow-hidden relative">
             <div className="p-4 border-b border-border flex justify-between items-center bg-surface-hover/30">
               <h3 className="font-bold text-foreground text-sm tracking-wide uppercase flex items-center gap-2">
                 Display Canvas
                 {currentChart && <span className="text-[10px] bg-emerald-500/10 text-emerald-500 px-2 py-0.5 rounded-full">RENDERED</span>}
               </h3>
               {currentChart && (
                 <span className="text-xs text-slate-400 font-medium">Interactive Plotly.js rendered plot</span>
               )}
             </div>

             <div className="flex-1 p-2 relative flex items-center justify-center bg-background/50">
               <AnimatePresence mode="wait">
                 {currentChart ? (
                   <motion.div 
                     key={currentKey}
                     initial={{ opacity: 0, scale: 0.98 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }}
                     className="w-full h-[550px] bg-surface rounded-xl overflow-hidden shadow-inner border border-border flex items-center justify-center"
                   >
                     <PlotlyChart figure={currentChart} height={550} />
                   </motion.div>
                 ) : (
                   <motion.div 
                     initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                     className="flex flex-col items-center justify-center text-slate-400 gap-4"
                   >
                     <div className="w-20 h-20 bg-surface-hover rounded-full flex items-center justify-center border border-border shadow-sm">
                        <MousePointerClick className="text-slate-300" size={32} />
                     </div>
                     <p className="text-sm font-medium tracking-wide">Configure chart limits and click <span className="text-foreground">Render Chart</span></p>
                   </motion.div>
                 )}
               </AnimatePresence>

               {/* Loading Overlay */}
               {currentLoading && (
                 <motion.div 
                   initial={{ opacity: 0 }} animate={{ opacity: 1 }}
                   className="absolute inset-0 bg-background/60 backdrop-blur-[2px] flex items-center justify-center z-10"
                 >
                   <div className="bg-surface px-6 py-4 rounded-2xl shadow-xl border border-border flex items-center gap-4">
                     <div className="w-6 h-6 border-2 border-brand-500 border-t-transparent rounded-full animate-spin" />
                     <span className="font-bold text-sm text-foreground">Computing Visualization...</span>
                   </div>
                 </motion.div>
               )}
             </div>
           </div>
        </div>

      </div>
    </div>
  )
}
