import { useState, useEffect, useRef } from 'react'
import { useSessionStore, useUIStore } from '../store'
import DropdownSelect from '../components/DropdownSelect'
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

  if (!session) return <div className="hyperopt-shell text-center">No session. <a href="/" className="text-brand-600 hover:text-brand-700">Upload →</a></div>
  if (!session.target_col) return <div className="hyperopt-shell text-center">No feature selection. <a href="/preprocess" className="text-brand-600 hover:text-brand-700">Preprocess →</a></div>

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

  const statusClass = (s: string) => ({
    complete: 'text-emerald-600',
    running: 'text-brand-600',
    queued: 'text-amber-600',
    failed: 'text-red-600',
  })[s] || 'text-slate-500'

  return (
    <div className="hyperopt-shell">
      <h1 className="hyperopt-title">Hyperparameter Tuning</h1>
      <p className="hyperopt-subtitle">Optuna TPE search — {session.feature_cols.length} features → {session.target_col} ({session.task_type})</p>

      <div className="hyperopt-grid">
        <div className="hyperopt-sidebar">
          <h3 className="hyperopt-sidebar-title">Configure Search</h3>
          <div className="hyperopt-field">
            <label className="hyperopt-label">Model</label>
            <DropdownSelect
              value={modelName}
              onChange={setModelName}
              ariaLabel="Model"
              options={MODELS.map(m => ({ value: m, label: m.replace(/_/g, ' ') }))}
              buttonClassName="w-full"
            />
          </div>
          <div className="hyperopt-field">
            <label className="hyperopt-label">Trials: {nTrials}</label>
            <input type="range" min={10} max={100} step={5} value={nTrials}
              onChange={e => setNTrials(+e.target.value)} className="hyperopt-range" aria-label="Trial count" />
            <div className="hyperopt-range-meta">
              <span>10</span><span>100</span>
            </div>
          </div>
          <div className="hyperopt-field">
            <label className="hyperopt-label">Timeout: {timeoutSecs}s</label>
            <input type="range" min={30} max={600} step={30} value={timeoutSecs}
              onChange={e => setTimeoutSecs(+e.target.value)} className="hyperopt-range" aria-label="Timeout seconds" />
          </div>
          <button onClick={startJob} disabled={loading} className="hyperopt-start-btn">
            {loading ? 'Starting…' : 'Start Search'}
          </button>

          {jobs.length > 0 && (
            <div className="hyperopt-jobs-wrap">
              <h4 className="hyperopt-jobs-title">Jobs</h4>
              {jobs.map(j => (
                <div key={j.job_id} onClick={() => setActiveJob(j)} className={cn('hyperopt-job', activeJob?.job_id === j.job_id && 'hyperopt-job--active')}>
                  <div className="hyperopt-job-title">{j.model_name}</div>
                  <div className="hyperopt-job-meta">
                    <span className={statusClass(j.status)}>{j.status}</span>
                    <span>{j.trials_done}/{j.n_trials}</span>
                  </div>
                  {j.status === 'running' && (
                    <progress className="hyperopt-progress" value={j.progress_pct || 0} max={100} />
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        <div>
          {activeJob && (
            <div className="hyperopt-main-card">
              <h3 className="hyperopt-main-title">
                {activeJob.model_name} — <span className={statusClass(activeJob.status)}>{activeJob.status}</span>
              </h3>
              <div className="hyperopt-stats">
                {[
                  { label: 'Trials Done', value: `${activeJob.trials_done} / ${activeJob.n_trials}` },
                  { label: 'Progress', value: `${activeJob.progress_pct ?? 0}%` },
                  { label: 'Best Score', value: activeJob.current_best != null ? (activeJob.current_best * 100).toFixed(2) + '%' : '—' },
                ].map(({ label, value }) => (
                  <div key={label} className="hyperopt-stat">
                    <div className="hyperopt-stat-value">{value}</div>
                    <div className="hyperopt-stat-label">{label}</div>
                  </div>
                ))}
              </div>
              {(activeJob.status === 'queued' || activeJob.status === 'running') && (
                <progress className="hyperopt-progress-lg w-full" value={activeJob.progress_pct ?? 0} max={100} />
              )}
            </div>
          )}

          {result && (
            <div className="hyperopt-result-grid">
              <div className="hyperopt-result-panel">
                <h3 className="hyperopt-result-title">Best Hyperparameters</h3>
                <div className="hyperopt-score">
                  <span className="hyperopt-score-value">
                    Best {result.metric}: {(result.best_score * 100).toFixed(3)}%
                  </span>
                  <div className="hyperopt-score-note">{result.n_trials} trials completed</div>
                </div>
                <div className="hyperopt-params">
                  {Object.entries(result.best_params || {}).map(([k, v]) => (
                    <div key={k} className="hyperopt-param-row">
                      <span className="hyperopt-param-key">{k}</span>
                      <span className="hyperopt-param-value">{String(v)}</span>
                    </div>
                  ))}
                  {Object.keys(result.best_params || {}).length === 0 && (
                    <p className="hyperopt-none">No tunable parameters for this model.</p>
                  )}
                </div>
              </div>

              <div className="hyperopt-train-panel">
                <div className="hyperopt-train-emoji">Ready</div>
                <div className="hyperopt-train-text">Ready to train with best params!</div>
                <div className="hyperopt-train-note">Go to Train and use these hyperparameters to get the optimised model.</div>
                <a href="/train" className="hyperopt-train-link">
                  → Train Optimised Model
                </a>
              </div>
            </div>
          )}

          {historyChart && (
            <div className="hyperopt-history-card">
              <h3 className="hyperopt-history-title">Optimization History</h3>
              <PlotlyChart figure={historyChart} height={350} />
            </div>
          )}

          {!activeJob && (
            <div className="hyperopt-empty-state">
              <div className="hyperopt-empty-state-emoji">Search</div>
              <div>Configure and start a hyperparameter search to find the best settings for your model.</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
