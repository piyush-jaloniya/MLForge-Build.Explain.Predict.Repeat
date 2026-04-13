import { useState, useEffect, useRef } from 'react'
import { useSessionStore, useUIStore } from '../store'
import PlotlyChart from '../components/PlotlyChart'
import api from '../api/client'

const MODELS = [
  'random_forest', 'gradient_boosting', 'xgboost', 'lightgbm',
  'logistic_regression', 'ridge', 'svm', 'decision_tree', 'knn',
]

export default function HyperoptPage() {
  const session = useSessionStore((s) => s.session)
  const notify = useUIStore((s) => s.notify)
  const [modelName, setModelName] = useState('random_forest')
  const [nTrials, setNTrials] = useState(30)
  const [timeoutSecs, setTimeoutSecs] = useState(120)
  const [jobs, setJobs] = useState<any[]>([])
  const [activeJob, setActiveJob] = useState<any>(null)
  const [result, setResult] = useState<any>(null)
  const [historyChart, setHistoryChart] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const pollRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})

  useEffect(() => {
    return () => {
      Object.values(pollRef.current).forEach(clearInterval)
      pollRef.current = {}
    }
  }, [])

  if (!session) return <div style={{ padding: '2rem', textAlign: 'center' }}>No session. <a href="/" style={{ color: '#4f46e5' }}>Upload →</a></div>
  if (!session.target_col) return <div style={{ padding: '2rem', textAlign: 'center' }}>No feature selection. <a href="/preprocess" style={{ color: '#4f46e5' }}>Preprocess →</a></div>

  const startJob = async () => {
    setLoading(true)
    try {
      const res = await api.post('/api/hyperopt/start', {
        session_id: session.session_id,
        model_name: modelName,
        n_trials: nTrials,
        timeout_seconds: timeoutSecs,
        feature_cols: session.feature_cols,
        target_col: session.target_col,
        task_type: session.task_type,
      })
      const jobId = res.data.job_id
      const newJob = { job_id: jobId, model_name: modelName, status: 'queued', trials_done: 0, n_trials: nTrials, current_best: null }
      setJobs(j => [newJob, ...j])
      setActiveJob(newJob)
      setResult(null)
      setHistoryChart(null)
      notify('info', `Hyperopt started: ${nTrials} trials for ${modelName}`)

      pollRef.current[jobId] = setInterval(async () => {
        try {
          const st = await api.get(`/api/hyperopt/status/${jobId}`)
          const d = st.data
          setJobs(j => j.map(j2 => j2.job_id === jobId ? { ...j2, ...d } : j2))
          setActiveJob((prev: any) => prev?.job_id === jobId ? { ...prev, ...d } : prev)

          if (d.status === 'complete') {
            clearInterval(pollRef.current[jobId]); delete pollRef.current[jobId]
            notify('success', `Hyperopt done! Best ${d.model_name}: ${d.current_best?.toFixed(4)}`)
            const rr = await api.get(`/api/hyperopt/results/${jobId}`)
            setResult(rr.data)
            const cc = await api.get(`/api/viz/hyperopt-history/${jobId}`)
            setHistoryChart(cc.data.chart)
          } else if (d.status === 'failed') {
            clearInterval(pollRef.current[jobId]); delete pollRef.current[jobId]
            notify('error', 'Hyperopt failed')
          }
        } catch { }
      }, 1500)
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'Failed to start')
    } finally {
      setLoading(false)
    }
  }

  const statusColor = (s: string) => ({ complete: '#059669', running: '#4f46e5', queued: '#d97706', failed: '#dc2626' })[s] || '#6b7280'

  return (
    <div style={{ padding: '1.5rem', maxWidth: 1100, margin: '0 auto' }}>
      <h1 style={{ fontSize: '1.5rem', fontWeight: 700, marginBottom: '0.25rem' }}>Hyperparameter Tuning</h1>
      <p style={{ color: '#6b7280', marginBottom: '1.5rem' }}>Optuna TPE search — {session.feature_cols.length} features → {session.target_col} ({session.task_type})</p>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '1.5rem' }}>
        <div style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 8, padding: '1.25rem' }}>
          <h3 style={{ fontWeight: 700, marginBottom: '1rem' }}>Configure Search</h3>
          <div style={{ marginBottom: '1rem' }}>
            <label style={lbl}>Model</label>
            <select value={modelName} onChange={e => setModelName(e.target.value)} style={sel}>
              {MODELS.map(m => <option key={m} value={m}>{m.replace(/_/g, ' ')}</option>)}
            </select>
          </div>
          <div style={{ marginBottom: '1rem' }}>
            <label style={lbl}>Trials: {nTrials}</label>
            <input type="range" min={10} max={100} step={5} value={nTrials}
              onChange={e => setNTrials(+e.target.value)} style={{ width: '100%' }} />
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#9ca3af' }}>
              <span>10</span><span>100</span>
            </div>
          </div>
          <div style={{ marginBottom: '1.25rem' }}>
            <label style={lbl}>Timeout: {timeoutSecs}s</label>
            <input type="range" min={30} max={600} step={30} value={timeoutSecs}
              onChange={e => setTimeoutSecs(+e.target.value)} style={{ width: '100%' }} />
          </div>
          <button onClick={startJob} disabled={loading} style={{
            width: '100%', padding: '0.875rem', background: '#4f46e5', color: 'white',
            border: 'none', borderRadius: 8, cursor: 'pointer', fontWeight: 700,
            opacity: loading ? 0.5 : 1,
          }}>
            {loading ? 'Starting…' : 'Start Search'}
          </button>

          {jobs.length > 0 && (
            <div style={{ marginTop: '1.5rem' }}>
              <h4 style={{ fontWeight: 700, marginBottom: '0.5rem', fontSize: '0.8rem', color: '#6b7280', textTransform: 'uppercase' }}>Jobs</h4>
              {jobs.map(j => (
                <div key={j.job_id} onClick={() => setActiveJob(j)} style={{
                  padding: '0.5rem', border: '1px solid', borderColor: activeJob?.job_id === j.job_id ? '#4f46e5' : '#e5e7eb',
                  borderRadius: 6, marginBottom: '0.4rem', cursor: 'pointer',
                  background: activeJob?.job_id === j.job_id ? '#eff6ff' : 'white',
                }}>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600 }}>{j.model_name}</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: '#6b7280' }}>
                    <span style={{ color: statusColor(j.status) }}>{j.status}</span>
                    <span>{j.trials_done}/{j.n_trials}</span>
                  </div>
                  {j.status === 'running' && (
                    <div style={{ height: 4, background: '#e5e7eb', borderRadius: 2, marginTop: '0.3rem' }}>
                      <div style={{ height: '100%', width: `${j.progress_pct || 0}%`, background: '#4f46e5', borderRadius: 2 }} />
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          {activeJob && (
            <div style={{ background: 'white', border: '1px solid #e5e7eb', borderRadius: 8, padding: '1.25rem', marginBottom: '1.25rem' }}>
              <h3 style={{ fontWeight: 700, marginBottom: '1rem' }}>
                {activeJob.model_name} — <span style={{ color: statusColor(activeJob.status) }}>{activeJob.status}</span>
              </h3>
              <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
                {[
                  { label: 'Trials Done', value: `${activeJob.trials_done} / ${activeJob.n_trials}` },
                  { label: 'Progress', value: `${activeJob.progress_pct ?? 0}%` },
                  { label: 'Best Score', value: activeJob.current_best != null ? (activeJob.current_best * 100).toFixed(2) + '%' : '—' },
                ].map(({ label, value }) => (
                  <div key={label} style={{ background: '#f9fafb', padding: '0.75rem 1rem', borderRadius: 8, textAlign: 'center', minWidth: 100 }}>
                    <div style={{ fontWeight: 700, fontSize: '1.25rem', color: '#4f46e5' }}>{value}</div>
                    <div style={{ fontSize: '0.7rem', color: '#6b7280' }}>{label}</div>
                  </div>
                ))}
              </div>
              {(activeJob.status === 'queued' || activeJob.status === 'running') && (
                <div style={{ height: 8, background: '#e5e7eb', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${activeJob.progress_pct ?? 0}%`, background: '#4f46e5', transition: 'width 0.5s', borderRadius: 4 }} />
                </div>
              )}
            </div>
          )}

          {result && (
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.25rem', marginBottom: '1.25rem' }}>
              <div style={{ background: 'white', border: '1px solid #e5e7eb', borderRadius: 8, padding: '1.25rem' }}>
                <h3 style={{ fontWeight: 700, marginBottom: '0.75rem' }}>Best Hyperparameters</h3>
                <div style={{ background: '#f0fdf4', padding: '0.75rem', borderRadius: 6, marginBottom: '0.75rem' }}>
                  <span style={{ fontWeight: 700, color: '#059669' }}>
                    Best {result.metric}: {(result.best_score * 100).toFixed(3)}%
                  </span>
                  <div style={{ fontSize: '0.75rem', color: '#6b7280' }}>{result.n_trials} trials completed</div>
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                  {Object.entries(result.best_params || {}).map(([k, v]) => (
                    <div key={k} style={{ display: 'flex', justifyContent: 'space-between', padding: '0.3rem 0.5rem', background: '#f9fafb', borderRadius: 4, fontSize: '0.8rem' }}>
                      <span style={{ color: '#6b7280' }}>{k}</span>
                      <span style={{ fontWeight: 600 }}>{String(v)}</span>
                    </div>
                  ))}
                  {Object.keys(result.best_params || {}).length === 0 && (
                    <p style={{ color: '#9ca3af', fontSize: '0.8rem' }}>No tunable parameters for this model.</p>
                  )}
                </div>
              </div>

              <div style={{ background: '#eff6ff', border: '1px solid #bfdbfe', borderRadius: 8, padding: '1.25rem', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', gap: '0.75rem' }}>
                <div style={{ fontSize: '2rem' }}>Ready</div>
                <div style={{ fontWeight: 700, textAlign: 'center' }}>Ready to train with best params!</div>
                <div style={{ fontSize: '0.8rem', color: '#6b7280', textAlign: 'center' }}>Go to Train and use these hyperparameters to get the optimised model.</div>
                <a href="/train" style={{ padding: '0.75rem 1.5rem', background: '#4f46e5', color: 'white', borderRadius: 8, fontWeight: 700, textDecoration: 'none', fontSize: '0.875rem' }}>
                  → Train Optimised Model
                </a>
              </div>
            </div>
          )}

          {historyChart && (
            <div style={{ background: 'white', border: '1px solid #e5e7eb', borderRadius: 8, padding: '1.25rem' }}>
              <h3 style={{ fontWeight: 700, marginBottom: '0.75rem' }}>Optimization History</h3>
              <PlotlyChart figure={historyChart} height={350} />
            </div>
          )}

          {!activeJob && (
            <div style={{ background: '#f9fafb', border: '1px solid #e5e7eb', borderRadius: 12, padding: '3rem', textAlign: 'center', color: '#9ca3af' }}>
              <div style={{ fontSize: '3rem', marginBottom: '0.5rem' }}>Search</div>
              <div>Configure and start a hyperparameter search to find the best settings for your model.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

const lbl: React.CSSProperties = { display: 'block', fontSize: '0.8rem', fontWeight: 600, marginBottom: '0.35rem', color: '#374151' }
const sel: React.CSSProperties = { width: '100%', padding: '0.4rem 0.5rem', border: '1px solid #d1d5db', borderRadius: 6, fontSize: '0.8rem', background: 'white' }
