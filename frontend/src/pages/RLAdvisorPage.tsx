import { useState } from 'react'
import { motion } from 'framer-motion'
import { useSessionStore, useUIStore } from '../store'
import api from '../api/client'
import { cn } from '../lib/utils'
import { Sparkles, BrainCircuit, Loader2, ArrowRight, Activity, Cpu } from 'lucide-react'

interface Recommendation {
  model_name: string
  expected_win_rate: number
  confidence: number
  uncertainty: number
  rationale: string
}

interface BanditState {
  task_type: string
  history_len: number
  alpha: Record<string, number>
  beta: Record<string, number>
}

export default function RLAdvisorPage() {
  const session = useSessionStore(s => s.session)
  const notify = useUIStore(s => s.notify)
  const [recs, setRecs] = useState<Recommendation[]>([])
  const [state, setState] = useState<BanditState | null>(null)
  const [loading, setLoading] = useState(false)
  const [taskType, setTaskType] = useState(session?.task_type || 'classification')

  if (!session) {
    return (
      <div className="min-h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-16 h-16 bg-brand-500/10 text-brand-500 rounded-full flex items-center justify-center mb-4">
          <Sparkles size={32} />
        </div>
        <h2 className="text-xl font-bold text-foreground">No dataset loaded</h2>
        <p className="text-slate-500 mb-6 mt-1 text-sm">Please upload a dataset to receive model recommendations.</p>
        <a href="/" className="px-6 py-2 bg-brand-500 hover:bg-brand-600 text-white rounded-xl font-medium transition-colors border border-transparent shadow-sm">
          Upload Dataset
        </a>
      </div>
    )
  }

  const loadRecs = async () => {
    setLoading(true)
    try {
      const [recRes, stateRes] = await Promise.all([
        api.get('/api/train/rl-recommend', { params: { session_id: session.session_id, task_type: taskType, top_k: 5 } }),
        api.get('/api/train/rl-state', { params: { session_id: session.session_id, task_type: taskType } }),
      ])
      setRecs(recRes.data.recommendations || [])
      setState(stateRes.data)
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Failed to load recommendations')
    } finally { setLoading(false) }
  }

  const rankColor = (i: number) => ['text-amber-500', 'text-slate-400', 'text-amber-700', 'text-brand-500', 'text-emerald-500'][i] || 'text-slate-500'
  const rankBg = (i: number) => ['bg-amber-500', 'bg-slate-400', 'bg-amber-700', 'bg-brand-500', 'bg-emerald-500'][i] || 'bg-slate-500'
  const rankBorder = (i: number) => ['border-amber-500', 'border-slate-400', 'border-amber-700', 'border-brand-500', 'border-emerald-500'][i] || 'border-slate-500'
  const rankEmoji = (i: number) => ['🥇', '🥈', '🥉', '4th', '5th'][i] || `${i + 1}th`

  const armEntries = state ? Object.keys(state.alpha).map(name => ({
    name,
    q_value: state.alpha[name] / (state.alpha[name] + (state.beta[name] || 1)),
    alpha: state.alpha[name],
    beta: state.beta[name] || 1,
    n_pulls: Math.round(state.alpha[name] + (state.beta[name] || 1)) - 2,
  })).sort((a, b) => b.q_value - a.q_value) : []
  const maxQ = armEntries.length > 0 ? Math.max(...armEntries.map(a => a.q_value)) : 1

  return (
    <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-6">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-foreground flex items-center gap-3 tracking-tight">
             <Sparkles className="text-brand-500" size={32} />
             RL Model Advisor
          </h1>
          <p className="text-slate-500 mt-2 font-medium max-w-2xl">
             Thompson Sampling bandit dynamically learns from your training results to recommend the best algorithms for your specific dataset.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="bg-surface border border-border rounded-2xl p-4 flex flex-wrap gap-4 items-center shadow-sm">
        <div className="flex items-center gap-3 whitespace-nowrap">
          <label className="text-xs font-bold uppercase tracking-wider text-slate-500">Task Type</label>
          <select 
            value={taskType} onChange={e => setTaskType(e.target.value)}
            className="px-4 py-2.5 bg-surface-hover border border-border rounded-xl text-sm font-medium focus:outline-none focus:border-brand-500 transition-all font-sans"
          >
            <option value="classification">Classification</option>
            <option value="regression">Regression</option>
          </select>
        </div>
        
        <button 
          onClick={loadRecs} disabled={loading} 
          className="flex items-center gap-2 px-6 py-2.5 bg-brand-500 hover:bg-brand-600 border border-brand-400 text-white rounded-xl text-sm font-bold transition-all shadow-sm active:scale-[0.98] disabled:opacity-50"
        >
          {loading ? <Loader2 size={18} className="animate-spin" /> : <BrainCircuit size={18} />}
          {loading ? 'Analyzing...' : 'Generate Recommendations'}
        </button>
        
        {session.feature_cols.length > 0 && (
          <div className="ml-auto flex items-center gap-2 px-4 py-2 bg-slate-100 dark:bg-slate-800 rounded-lg text-xs font-semibold text-slate-500 uppercase tracking-widest">
            <span>{session.n_rows.toLocaleString()} Rows</span>
            <span className="w-1 h-1 rounded-full bg-slate-400" />
            <span className="text-brand-500">{session.feature_cols.length} Features</span>
            {session.target_col && (
              <>
                <span className="w-1 h-1 rounded-full bg-slate-400" />
                <span className="text-emerald-500">{session.target_col} target</span>
              </>
            )}
          </div>
        )}
      </div>

      {recs.length > 0 && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
          
          {/* Top Recommendations */}
          <div>
            <h2 className="text-lg font-bold text-foreground mb-4 flex items-center gap-2">
              <Activity size={20} className="text-brand-500" /> Top Recommendations
            </h2>
            <div className="space-y-4">
              {recs.map((rec, i) => (
                <motion.div 
                  initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.1 }}
                  key={rec.model_name} 
                  className={cn(
                    "border rounded-2xl p-5 shadow-sm transition-all group",
                    i === 0 
                      ? "bg-amber-500/5 border-amber-500/30 dark:border-amber-500/20" 
                      : "bg-surface border-border hover:border-brand-500/30 hover:shadow-md"
                  )}
                >
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl drop-shadow-sm">{rankEmoji(i)}</span>
                      <span className="font-bold text-lg text-foreground tracking-tight">
                        {rec.model_name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())}
                      </span>
                    </div>
                    <div className="text-right flex flex-col items-end">
                      <div className={cn("text-sm font-bold tracking-tight", rankColor(i))}>
                        E[Win Rate]: {(rec.expected_win_rate * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs font-semibold text-slate-400 mt-0.5">
                        Confidence: {(rec.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  </div>
                  
                  <p className="text-sm text-foreground/80 leading-relaxed mb-4">{rec.rationale}</p>
                  
                  <div className="h-1.5 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                    <motion.div 
                      initial={{ width: 0 }} animate={{ width: `${rec.expected_win_rate * 100}%` }} transition={{ duration: 1, delay: 0.2 }}
                      className={cn("h-full rounded-full", rankBg(i))}
                    />
                  </div>
                  
                  {i === 0 && (
                    <div className="mt-4 pt-4 border-t border-amber-500/10">
                      <a href="/train" className="inline-flex items-center gap-2 px-4 py-2 bg-amber-500 hover:bg-amber-600 text-white rounded-lg text-xs font-bold transition-all shadow-sm active:scale-95">
                        Train this model <ArrowRight size={14} />
                      </a>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          </div>

          {/* Bandit Posterior State */}
          {state && armEntries.length > 0 && (
            <div>
              <h2 className="text-lg font-bold text-foreground mb-4 flex items-center gap-2">
                <Cpu size={20} className="text-brand-500" /> 
                Bandit Posterior
                <span className="ml-2 px-2 py-0.5 bg-brand-500/10 text-brand-500 rounded text-xs font-semibold uppercase tracking-wider">
                  {state.history_len} recorded outcomes
                </span>
              </h2>
              
              <div className="bg-surface border border-border rounded-2xl overflow-hidden shadow-sm">
                <div className="grid grid-cols-[140px_1fr_50px_50px] gap-4 px-5 py-3 bg-surface-hover/50 border-b border-border text-[10px] font-bold text-slate-500 uppercase tracking-widest text-left">
                  <span>Model</span><span>Posterior Mean Win Rate</span><span className="text-center">α</span><span className="text-center">β</span>
                </div>
                
                <div className="divide-y divide-border/50">
                  {armEntries.map(arm => (
                    <div key={arm.name} className="grid grid-cols-[140px_1fr_50px_50px] gap-4 px-5 py-3 items-center hover:bg-surface-hover/30 transition-colors">
                      <div className="text-xs font-bold text-foreground truncate">
                        {arm.name.replace(/_/g, ' ')}
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="flex-1 h-2 bg-slate-100 dark:bg-slate-800 rounded-full overflow-hidden">
                          <motion.div 
                            initial={{ width: 0 }} animate={{ width: `${(arm.q_value / maxQ) * 100}%` }} transition={{ duration: 1 }}
                            className="h-full bg-brand-500 rounded-full"
                          />
                        </div>
                        <span className="text-xs font-bold text-brand-500 min-w-[36px]">
                          {(arm.q_value * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-xs font-semibold text-slate-500 text-center">{arm.alpha.toFixed(1)}</div>
                      <div className="text-xs font-semibold text-slate-500 text-center">{arm.beta.toFixed(1)}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="mt-4 p-5 bg-emerald-500/5 border border-emerald-500/20 rounded-2xl">
                <h4 className="text-xs font-bold text-emerald-600 dark:text-emerald-400 uppercase tracking-wider mb-2">How it works</h4>
                <p className="text-xs text-foreground/70 leading-relaxed">
                  Each model maintains a Beta(α, β) prior distribution. α increases on successful training outcomes (high metric), while β increases on poor outcomes. The mean α/(α+β) represents the estimated win rate. Thompson Sampling probabilistically selects models by drawing from these distributions—perfectly balancing exploration of new models with exploitation of known good ones.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {recs.length === 0 && !loading && (
        <div className="bg-surface border-2 border-dashed border-border rounded-3xl p-16 flex flex-col items-center text-center">
          <div className="w-24 h-24 bg-brand-500/5 rounded-full flex items-center justify-center mb-6">
             <BrainCircuit size={48} className="text-brand-500/50" strokeWidth={1} />
          </div>
          <h2 className="text-2xl font-bold text-foreground mb-2">Thompson Sampling Bandit</h2>
          <p className="text-slate-500 max-w-lg mb-8 leading-relaxed">
             The intelligent advisor uses Bayesian Beta-Bernoulli bandits to learn exactly which standard machine learning algorithms perform best for your dataset characteristics. It automatically updates its knowledge base after every training run you perform.
          </p>
          <button 
             onClick={loadRecs} 
             className="px-6 py-3 bg-brand-500 hover:bg-brand-600 text-white font-bold rounded-xl shadow-md shadow-brand-500/20 transition-all flex items-center gap-2"
          >
             <Sparkles size={18} /> Initialize Analysis
          </button>
        </div>
      )}
    </div>
  )
}
