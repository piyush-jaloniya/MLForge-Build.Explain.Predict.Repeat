import { useState, useRef, useEffect, type ReactNode } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useLocation } from 'react-router-dom'
import { useSessionStore, useModelStore, useUIStore } from '../store'
import api from '../api/client'
import {
  Activity,
  Bot,
  Brain,
  ChevronRight,
  Copy,
  Database,
  Files,
  Layers3,
  ListChecks,
  Loader2,
  Send,
  Sparkles,
  TableProperties,
  Target,
  Wand2,
  X,
} from 'lucide-react'
import { cn } from '../lib/utils'

interface Message { role: 'user' | 'assistant'; content: string; ts: number }
interface InsightState { content: string; ok: boolean | null; error: string | null }

const EMPTY_INSIGHT: InsightState = { content: '', ok: null, error: null }

const STARTERS = [
  'What preprocessing steps should I apply to this dataset?',
  'Which model should I use for classification?',
  "How do I interpret my model's accuracy?",
  'What does feature importance mean?',
]

function renderInlineMarkdown(text: string) {
  return text.split(/(\*\*.*?\*\*|__.*?__)/g).map((part, index) => {
    const isBold = (
      (part.startsWith('**') && part.endsWith('**')) ||
      (part.startsWith('__') && part.endsWith('__'))
    )
    if (!isBold) return part
    return <strong key={index} className="font-semibold text-current">{part.slice(2, -2)}</strong>
  })
}

function renderRichText(text: string) {
  return text.split('\n').map((rawLine, index) => {
    const line = rawLine.trimEnd()

    if (line.trim() === '') {
      return <div key={index} className="h-3" />
    }

    const headingMatch = line.match(/^(#{1,3})\s+(.*)$/)
    if (headingMatch) {
      const level = headingMatch[1].length
      const content = headingMatch[2]
      const headingClass = level === 1
        ? 'text-base font-bold text-foreground'
        : level === 2
          ? 'text-sm font-bold text-foreground'
          : 'text-sm font-semibold text-foreground'
      return <div key={index} className={headingClass}>{renderInlineMarkdown(content)}</div>
    }

    const numberedMatch = line.match(/^(\d+)\.\s+(.*)$/)
    if (numberedMatch) {
      return (
        <div key={index} className="flex gap-3 text-sm leading-6 text-slate-700 dark:text-slate-200">
          <span className="mt-0.5 min-w-5 font-bold text-brand-600 dark:text-brand-300">{numberedMatch[1]}.</span>
          <span>{renderInlineMarkdown(numberedMatch[2])}</span>
        </div>
      )
    }

    const bulletMatch = line.match(/^[-*]\s+(.*)$/)
    if (bulletMatch) {
      return (
        <div key={index} className="flex gap-3 text-sm leading-6 text-slate-700 dark:text-slate-200">
          <span className="mt-[0.6rem] h-1.5 w-1.5 rounded-full bg-brand-500 dark:bg-brand-300" />
          <span>{renderInlineMarkdown(bulletMatch[1])}</span>
        </div>
      )
    }

    return (
      <p key={index} className="text-sm leading-6 text-slate-700 dark:text-slate-200">
        {renderInlineMarkdown(line)}
      </p>
    )
  })
}

function formatTime(ts: number) {
  return new Intl.DateTimeFormat(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  }).format(ts)
}

function truncateText(text: string, max = 3) {
  const parts = text.split(/\s+/)
  if (parts.length <= max) return text
  return `${parts.slice(0, max).join(' ')}...`
}

function getPagePresets(pathname: string, context: {
  filename?: string | null
  targetCol?: string | null
  modelName?: string | null
}) {
  const { filename, targetCol, modelName } = context

  if (pathname === '/preprocess') {
    return {
      label: 'Preprocess Presets',
      prompts: [
        `Suggest a cleaning plan for ${filename || 'this dataset'}.`,
        'Which preprocessing steps should I apply first and in what order?',
        'What should I scale, encode, drop, or leave untouched in this dataset?',
      ],
    }
  }

  if (pathname === '/train') {
    return {
      label: 'Train Presets',
      prompts: [
        `Recommend the top 3 models for ${targetCol || 'this target'} and explain why.`,
        'What model should I start with for a strong baseline here?',
        `What hyperparameters matter most for ${modelName ? modelName.replace(/_/g, ' ') : 'the selected model'}?`,
      ],
    }
  }

  if (pathname === '/metrics') {
    return {
      label: 'Metrics Presets',
      prompts: [
        `Explain whether ${modelName ? modelName.replace(/_/g, 'this model') : 'this model'} is actually good.`,
        'Which metrics matter most here and how should I interpret them?',
        'What should I improve next based on these results?',
      ],
    }
  }

  if (pathname === '/viz') {
    return {
      label: 'Visualization Presets',
      prompts: [
        'Which charts should I look at first to understand this dataset?',
        'What patterns or anomalies should I try to confirm visually?',
        'How should I use visualization findings to guide preprocessing?',
      ],
    }
  }

  if (pathname === '/quality') {
    return {
      label: 'Quality Presets',
      prompts: [
        'Summarize the biggest data quality risks in this dataset.',
        'Which quality issues matter most before training?',
        'Give me a priority order for fixing the detected data quality problems.',
      ],
    }
  }

  if (pathname === '/xai') {
    return {
      label: 'Explainability Presets',
      prompts: [
        'What do these explanations say about how the model is making decisions?',
        'Which features are helping or hurting the model most?',
        'Are there any explainability red flags I should worry about?',
      ],
    }
  }

  if (pathname === '/report') {
    return {
      label: 'Report Presets',
      prompts: [
        'Draft a concise executive summary from the current experiment.',
        'Turn the current findings into stakeholder-friendly language.',
        'What recommendations should go into the final report?',
      ],
    }
  }

  return {
    label: 'Suggested Prompts',
    prompts: STARTERS,
  }
}

export default function AIAssistantDrawer() {
  const location = useLocation()
  const session = useSessionStore(s => s.session)
  const runs = useModelStore(s => s.runs)
  const activeRunId = useModelStore(s => s.activeRunId)
  const darkMode = useUIStore(s => s.darkMode)
  const notify = useUIStore(s => s.notify)
  const isAiDrawerOpen = useUIStore(s => s.isAiDrawerOpen)
  const setAiDrawerOpen = useUIStore(s => s.setAiDrawerOpen)

  const [messages, setMessages] = useState<Message[]>([{
    role: 'assistant',
    content: "Hi! I'm your ML assistant. I can help you understand your data, choose the right model, interpret results, and debug issues. What would you like to know?",
    ts: Date.now(),
  }])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [schemaInsight, setSchemaInsight] = useState<InsightState>(EMPTY_INSIGHT)
  const [modelRec, setModelRec] = useState<InsightState>(EMPTY_INSIGHT)
  const [metricsInsight, setMetricsInsight] = useState<InsightState>(EMPTY_INSIGHT)
  const [loadingInsight, setLoadingInsight] = useState('')
  const [isDesktopLayout, setIsDesktopLayout] = useState(() => window.innerWidth >= 1024)
  const bottomRef = useRef<HTMLDivElement>(null)

  const completedRuns = runs.filter(r => r.status === 'complete')
  const currentRunId = activeRunId || completedRuns[0]?.run_id
  const currentModelName = runs.find(r => r.run_id === currentRunId)?.model_name
  const showStarterPrompts = messages.length <= 2
  const datasetColumns = session?.column_names || []
  const featureCount = session?.feature_cols?.length || 0
  const schemaDisabledReason = session ? null : 'Select or upload a dataset first.'
  const modelDisabledReason = !session
    ? 'Select or upload a dataset first.'
    : !session.target_col
      ? 'Select a target column first.'
      : null
  const metricsDisabledReason = !currentRunId ? 'Train or select a completed model run first.' : null
  const currentPresetSection = getPagePresets(location.pathname, {
    filename: session?.filename,
    targetCol: session?.target_col,
    modelName: currentModelName,
  })

  useEffect(() => {
    if (isAiDrawerOpen) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, loading, isAiDrawerOpen])

  useEffect(() => {
    const onResize = () => {
      const desktop = window.innerWidth >= 1024
      setIsDesktopLayout(desktop)
    }

    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  const send = async (text: string) => {
    if (!text.trim() || loading) return
    const userMsg: Message = { role: 'user', content: text.trim(), ts: Date.now() }
    setMessages(m => [...m, userMsg])
    setInput('')
    setLoading(true)
    try {
      const res = await api.post('/api/ai/chat', {
        session_id: session?.session_id,
        run_id: currentRunId,
        message: text,
        history: messages.slice(-6).map(m => ({ role: m.role, content: m.content })),
      })
      const reply = res.data.ok === false
        ? `AI request failed: ${res.data.error || res.data.response || 'Unknown error'}`
        : res.data.response
      setMessages(m => [...m, { role: 'assistant', content: reply, ts: Date.now() }])
      if (res.data.ok === false) {
        notify('error', res.data.error || 'AI chat unavailable')
      }
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || 'AI chat unavailable')
    } finally {
      setLoading(false)
    }
  }

  const loadInsight = async (type: 'schema' | 'model' | 'metrics') => {
    setLoadingInsight(type)
    try {
      if (type === 'schema') {
        const r = await api.get(`/api/ai/schema/${session?.session_id}`)
        setSchemaInsight({
          content: r.data.ok === false ? (r.data.error || r.data.narrative) : r.data.narrative,
          ok: r.data.ok ?? true,
          error: r.data.error ?? null,
        })
        if (r.data.ok === false) notify('error', r.data.error || 'Dataset insight failed')
      } else if (type === 'model') {
        const r = await api.get(`/api/ai/recommend-model/${session?.session_id}`, {
          params: { task_type: session?.task_type || 'classification', target_col: session?.target_col }
        })
        setModelRec({
          content: r.data.ok === false ? (r.data.error || r.data.recommendation) : r.data.recommendation,
          ok: r.data.ok ?? true,
          error: r.data.error ?? null,
        })
        if (r.data.ok === false) notify('error', r.data.error || 'Model recommendation failed')
      } else if (type === 'metrics') {
        const r = await api.get(`/api/ai/interpret-metrics/${currentRunId}`)
        setMetricsInsight({
          content: r.data.ok === false ? (r.data.error || r.data.interpretation) : r.data.interpretation,
          ok: r.data.ok ?? true,
          error: r.data.error ?? null,
        })
        if (r.data.ok === false) notify('error', r.data.error || 'Metrics interpretation failed')
      }
    } catch (e: any) {
      notify('error', e?.response?.data?.detail || `Failed to load ${type} insight`)
    } finally {
      setLoadingInsight('')
    }
  }

  const copyText = async (value: string, successMessage: string) => {
    try {
      await navigator.clipboard.writeText(value)
      notify('success', successMessage)
    } catch {
      notify('error', 'Clipboard copy failed')
    }
  }

  const applyInsightToChat = (title: string, content: string) => {
    setInput(
      `Using the ${title.toLowerCase()} below as context:\n${content}\n\nTurn this into concrete next steps for me.`
    )
  }

  const runMessageAction = (mode: 'simpler' | 'steps', content: string) => {
    if (mode === 'simpler') {
      send(`Explain this more simply for a beginner:\n\n${content}`)
      return
    }
    send(`Turn this into an actionable checklist with short steps:\n\n${content}`)
  }

  const progressItems = [
    {
      label: 'Dataset loaded',
      detail: session?.filename || 'Waiting for dataset',
      complete: !!session,
    },
    {
      label: 'Target selected',
      detail: session?.target_col || 'Target column not selected',
      complete: !!session?.target_col,
    },
    {
      label: 'Model trained',
      detail: currentModelName ? currentModelName.replace(/_/g, ' ') : 'No completed run yet',
      complete: !!currentRunId,
    },
    {
      label: 'Metrics ready',
      detail: currentRunId ? 'Interpretation available' : 'Train a model to unlock metrics insight',
      complete: !!currentRunId,
    },
  ]

  return (
    <AnimatePresence>
      {isAiDrawerOpen && (
        <motion.section
          initial={{ opacity: 0, scale: 0.985 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.99 }}
          transition={{ duration: 0.2 }}
          className={cn(
            'fixed inset-0 z-50 overflow-hidden',
            darkMode
              ? 'bg-[radial-gradient(circle_at_top_left,#1e1b4b_0%,#0f172a_48%,#020617_100%)] text-slate-100'
              : 'bg-[linear-gradient(135deg,#eef2ff_0%,#f8fafc_42%,#e2e8f0_100%)] text-slate-900'
          )}
        >
          <div className="pointer-events-none absolute inset-0 overflow-hidden">
            <div className={cn('absolute -left-20 top-0 h-72 w-72 rounded-full blur-3xl', darkMode ? 'bg-brand-500/15' : 'bg-brand-300/20')} />
            <div className={cn('absolute right-0 top-8 h-80 w-80 rounded-full blur-3xl', darkMode ? 'bg-cyan-500/10' : 'bg-cyan-200/20')} />
            <div className={cn('absolute bottom-0 left-1/3 h-72 w-72 rounded-full blur-3xl', darkMode ? 'bg-brand-700/10' : 'bg-amber-100/25')} />
          </div>

          <div className="relative flex h-full flex-col overflow-hidden">
            <header className={cn(
              'border-b px-4 py-2.5 backdrop-blur-xl',
              darkMode
                ? 'border-white/8 bg-slate-950/45'
                : 'border-white/60 bg-white/72'
            )}>
              <div className="flex items-center justify-between gap-4">
                <div className="flex min-w-0 items-center gap-4">
                  <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-2xl bg-gradient-to-br from-brand-500 via-brand-600 to-brand-700 text-white shadow-[0_10px_30px_rgba(79,70,229,0.35)]">
                    <Bot size={20} />
                  </div>
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-3">
                      <h2 className="text-[1.4rem] font-black tracking-tight text-foreground">AI Assistant Workspace</h2>
                      <span className="rounded-full border border-emerald-200 bg-emerald-50 px-3 py-1 text-[11px] font-black uppercase tracking-[0.2em] text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300">
                        Live
                      </span>
                    </div>
                    <p className="mt-0.5 hidden text-sm leading-6 text-foreground/75 md:block">
                      Explore structured insights on the left and keep the conversation flowing on the right.
                    </p>
                  </div>
                </div>

                <button
                  onClick={() => setAiDrawerOpen(false)}
                  className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-border/70 bg-surface/85 text-foreground/70 shadow-sm transition-all hover:-translate-y-0.5 hover:border-brand-400 hover:text-brand-600"
                  aria-label="Close AI assistant"
                >
                  <X size={18} />
                </button>
              </div>
            </header>

            <div
              className={cn(
                'flex min-h-0 flex-1 gap-3 overflow-hidden p-3 md:p-4',
                !isDesktopLayout && 'flex-col overflow-y-auto'
              )}
            >
              <aside
                className={cn(
                  'min-h-0 rounded-[32px] border p-4 shadow-[0_16px_45px_rgba(15,23,42,0.08)] backdrop-blur-xl lg:w-[430px] xl:w-[470px]',
                  darkMode
                    ? 'border-white/8 bg-slate-950/50'
                    : 'border-white/65 bg-white/74'
                )}
              >
                <div className="flex h-full min-h-0 flex-col">
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <div>
                      <p className="text-[11px] font-black uppercase tracking-[0.24em] text-brand-600 dark:text-brand-300">Insight Studio</p>
                      <h3 className="mt-1 text-lg font-bold text-foreground">Workspace context and fast analysis</h3>
                    </div>
                    <span className="rounded-full border border-brand-200 bg-brand-50 px-3 py-1 text-[11px] font-bold text-brand-700 dark:border-brand-900/60 dark:bg-brand-950/40 dark:text-brand-300">
                      {loadingInsight ? 'Analyzing' : 'Ready'}
                    </span>
                  </div>

                  <div className="min-h-0 flex-1 overflow-y-auto pr-1 custom-scrollbar">
                    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
                      <ContextChip
                        icon={<Database size={16} />}
                        label="Dataset"
                        value={session?.filename || 'No dataset loaded'}
                        tone="emerald"
                      />
                      <ContextChip
                        icon={<Target size={16} />}
                        label="Target"
                        value={session?.target_col || 'Choose a target column'}
                        tone="amber"
                      />
                      <ContextChip
                        icon={<Brain size={16} />}
                        label="Model"
                        value={currentModelName ? currentModelName.replace(/_/g, ' ') : 'No trained model selected'}
                        tone="brand"
                      />
                    </div>

                    <div className="mt-4 grid gap-3 sm:grid-cols-2">
                      <MiniStat
                        icon={<Files size={15} />}
                        label="Rows x Columns"
                        value={session ? `${session.n_rows.toLocaleString()} x ${session.n_cols}` : 'No dataset'}
                      />
                      <MiniStat
                        icon={<Layers3 size={15} />}
                        label="Task Type"
                        value={session?.task_type ? truncateText(session.task_type, 2) : 'Not selected'}
                      />
                      <MiniStat
                        icon={<Sparkles size={15} />}
                        label="Feature Count"
                        value={featureCount ? `${featureCount} selected` : 'Not selected'}
                      />
                      <MiniStat
                        icon={<TableProperties size={15} />}
                        label="Column Snapshot"
                        value={datasetColumns.length ? `${datasetColumns.length} columns` : 'No columns'}
                      />
                    </div>

                    <div className="mt-4 rounded-[28px] border border-border/70 bg-surface/80 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-[11px] font-black uppercase tracking-[0.22em] text-foreground/60">Dataset Snapshot</p>
                          <h4 className="mt-1 text-sm font-semibold text-foreground">Detected columns</h4>
                        </div>
                        <span className="rounded-full bg-surface-hover px-2.5 py-1 text-[11px] font-semibold text-foreground/75">
                          {datasetColumns.length}
                        </span>
                      </div>

                      <div className="mt-3 flex max-h-32 flex-wrap gap-2 overflow-auto custom-scrollbar">
                        {datasetColumns.length > 0 ? datasetColumns.slice(0, 16).map(col => (
                          <span
                            key={col}
                            className="rounded-full border border-border bg-surface px-2.5 py-1 text-xs font-medium text-foreground/80 shadow-sm"
                          >
                            {col}
                          </span>
                        )) : (
                          <p className="text-sm text-foreground/60">Upload a dataset to populate schema context here.</p>
                        )}
                      </div>
                    </div>

                    <div className="mt-4 rounded-[28px] border border-border/70 bg-surface/80 p-4">
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <p className="text-[11px] font-black uppercase tracking-[0.22em] text-foreground/60">Workflow Pulse</p>
                          <h4 className="mt-1 text-sm font-semibold text-foreground">Current ML progress</h4>
                        </div>
                        <span className="rounded-full bg-surface-hover px-2.5 py-1 text-[11px] font-semibold text-foreground/75">
                          {progressItems.filter(item => item.complete).length}/{progressItems.length}
                        </span>
                      </div>

                      <div className="mt-4 space-y-3">
                        {progressItems.map((item, index) => (
                          <div key={item.label} className="flex gap-3">
                            <div className="flex flex-col items-center">
                              <div className={cn(
                                'h-3.5 w-3.5 rounded-full border-2',
                                item.complete
                                  ? 'border-emerald-500 bg-emerald-500'
                                  : 'border-border bg-surface'
                              )} />
                              {index < progressItems.length - 1 && (
                                <div className={cn(
                                  'mt-1 h-7 w-px',
                                  item.complete ? 'bg-emerald-300 dark:bg-emerald-700/70' : 'bg-border'
                                )} />
                              )}
                            </div>
                            <div className="pb-1">
                              <p className="text-sm font-semibold text-foreground">{item.label}</p>
                              <p className="text-xs leading-5 text-foreground/70">{item.detail}</p>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="mt-4 grid gap-3">
                      <InsightCard
                        title="Dataset Insight"
                        description="Summarize structure, quality signals, and likely ML use cases."
                        icon={<Sparkles size={17} />}
                        content={schemaInsight.content}
                        ok={schemaInsight.ok}
                        loading={loadingInsight === 'schema'}
                        onClick={() => loadInsight('schema')}
                        disabled={!!schemaDisabledReason}
                        disabledReason={schemaDisabledReason}
                        onUse={() => applyInsightToChat('Dataset Insight', schemaInsight.content)}
                      />
                      <InsightCard
                        title="Model Recommendation"
                        description="Recommend promising algorithms from the current dataset setup."
                        icon={<Brain size={17} />}
                        content={modelRec.content}
                        ok={modelRec.ok}
                        loading={loadingInsight === 'model'}
                        onClick={() => loadInsight('model')}
                        disabled={!!modelDisabledReason}
                        disabledReason={modelDisabledReason}
                        onUse={() => applyInsightToChat('Model Recommendation', modelRec.content)}
                      />
                      <InsightCard
                        title="Metrics Interpretation"
                        description="Translate the latest evaluation metrics into plain language."
                        icon={<Activity size={17} />}
                        content={metricsInsight.content}
                        ok={metricsInsight.ok}
                        loading={loadingInsight === 'metrics'}
                        onClick={() => loadInsight('metrics')}
                        disabled={!!metricsDisabledReason}
                        disabledReason={metricsDisabledReason}
                        onUse={() => applyInsightToChat('Metrics Interpretation', metricsInsight.content)}
                      />
                    </div>
                  </div>
                </div>
              </aside>

              <section className={cn(
                'min-h-0 flex-1 rounded-[32px] border shadow-[0_18px_50px_rgba(15,23,42,0.1)] backdrop-blur-xl',
                darkMode
                  ? 'border-white/8 bg-slate-950/55'
                  : 'border-slate-200/70 bg-white/78'
              )}>
                <div className="flex h-full min-h-0 flex-col">
                  <div className="border-b border-border/70 px-5 py-3">
                    <div className="flex items-center justify-between gap-3">
                      <div>
                        <h3 className="text-base font-bold text-foreground">Conversation</h3>
                        <p className="mt-0.5 text-sm text-foreground/70">Ask anything about the current workflow</p>
                      </div>
                      <span className="rounded-full border border-border/70 bg-surface px-3 py-1 text-[11px] font-semibold text-foreground/75">
                        {Math.max(messages.length - 1, 0)} turns
                      </span>
                    </div>

                    {showStarterPrompts && (
                      <div className="mt-3">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <p className="text-[11px] font-black uppercase tracking-[0.18em] text-brand-600 dark:text-brand-300">
                            Quick prompts
                          </p>
                          <p className="text-[11px] font-medium text-foreground/70">
                            {currentPresetSection.label}
                          </p>
                        </div>
                        <div className="flex gap-2 overflow-x-auto pb-1 custom-scrollbar">
                          {currentPresetSection.prompts.map((starter, index) => (
                            <button
                              key={`${currentPresetSection.label}-${index}`}
                              onClick={() => send(starter)}
                              className="shrink-0 rounded-full border border-brand-200 bg-brand-50 px-3 py-1.5 text-xs font-semibold text-brand-700 transition-all hover:-translate-y-0.5 hover:border-brand-400/60 hover:bg-surface dark:border-brand-800 dark:bg-brand-950/50 dark:text-brand-200"
                            >
                              {starter}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4 custom-scrollbar">
                    <div className="flex flex-col gap-4">
                      {messages.map((msg, i) => (
                        <MessageBubble
                          key={i}
                          message={msg}
                          onCopy={copyText}
                          onAction={runMessageAction}
                        />
                      ))}

                      {loading && (
                        <div className="flex justify-start">
                          <div className="flex max-w-[88%] items-end gap-3">
                            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-2xl bg-brand-500/12 text-brand-600 dark:bg-brand-500/20 dark:text-brand-300">
                              <Bot size={16} />
                            </div>
                            <div className="rounded-[24px] rounded-bl-md border border-brand-100 bg-surface px-4 py-3 text-sm text-foreground/80 shadow-sm dark:border-brand-900/60 dark:bg-slate-900 dark:text-slate-200">
                              <div className="flex items-center gap-2">
                                <Loader2 size={15} className="animate-spin text-brand-500" />
                                <span>Thinking through your workflow...</span>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                    <div ref={bottomRef} className="h-1" />
                  </div>

                  <footer className="border-t border-border/70 px-5 py-3">
                    <div className="rounded-[24px] border border-border/70 bg-surface/90 p-2.5 shadow-[0_10px_30px_rgba(15,23,42,0.06)]">
                      <div className="flex gap-3">
                        <textarea
                          value={input}
                          onChange={e => setInput(e.target.value)}
                          onKeyDown={e => {
                            if (e.key === 'Enter' && !e.shiftKey) {
                              e.preventDefault()
                              send(input)
                            }
                          }}
                          placeholder="Ask about your data, models, metrics, or next best step..."
                          className="min-h-[62px] flex-1 resize-none bg-transparent px-1 py-1 text-sm leading-6 text-foreground outline-none placeholder:text-slate-400"
                          rows={Math.min(5, Math.max(2, input.split('\n').length))}
                        />

                        <button
                          onClick={() => send(input)}
                          disabled={loading || !input.trim()}
                          className="flex h-14 shrink-0 items-center justify-center gap-2 rounded-2xl bg-gradient-to-r from-brand-500 to-brand-600 px-5 text-sm font-semibold text-white shadow-lg shadow-brand-500/20 transition-all hover:-translate-y-0.5 hover:from-brand-600 hover:to-brand-700 disabled:translate-y-0 disabled:cursor-not-allowed disabled:opacity-55"
                        >
                          <Send size={17} />
                          <span>Send</span>
                        </button>
                      </div>

                      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 border-t border-border/60 pt-3 text-[11px] font-medium">
                        <span className="text-foreground/70">Press Enter to send. Use Shift+Enter for a new line.</span>
                        <div className="flex flex-wrap items-center gap-2 text-foreground/75">
                          {session && (
                            <span className="rounded-full bg-surface-hover px-2.5 py-1">
                              {session.filename}
                            </span>
                          )}
                          {session?.target_col && (
                            <span className="rounded-full bg-brand-50 px-2.5 py-1 text-brand-700 dark:bg-brand-950/50 dark:text-brand-300">
                              Target: {session.target_col}
                            </span>
                          )}
                          {currentModelName && (
                            <span className="rounded-full bg-emerald-50 px-2.5 py-1 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
                              {currentModelName.replace(/_/g, ' ')}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>
                  </footer>
                </div>
              </section>
            </div>
          </div>
        </motion.section>
      )}
    </AnimatePresence>
  )
}

function ContextChip({
  icon,
  label,
  value,
  tone,
}: {
  icon: ReactNode
  label: string
  value: string
  tone: 'emerald' | 'amber' | 'brand'
}) {
  const tones = {
    emerald: 'border-emerald-200/80 bg-emerald-50/85 text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/30 dark:text-emerald-300',
    amber: 'border-amber-200/80 bg-amber-50/85 text-amber-700 dark:border-amber-900/60 dark:bg-amber-950/30 dark:text-amber-300',
    brand: 'border-brand-200/80 bg-brand-50/85 text-brand-700 dark:border-brand-900/60 dark:bg-brand-950/40 dark:text-brand-300',
  }

  return (
    <div className={cn('rounded-2xl border px-3 py-3 shadow-sm', tones[tone])}>
      <div className="flex items-start gap-2">
        <div className="mt-0.5">{icon}</div>
        <div className="min-w-0">
          <p className="text-[10px] font-black uppercase tracking-[0.18em] opacity-70">{label}</p>
          <p className="mt-1 truncate text-sm font-semibold">{value}</p>
        </div>
      </div>
    </div>
  )
}

function MiniStat({
  icon,
  label,
  value,
}: {
  icon: ReactNode
  label: string
  value: string
}) {
  return (
    <div className="rounded-2xl border border-border/70 bg-surface/80 px-3 py-3 shadow-sm">
      <div className="flex items-center gap-2 text-foreground/60">
        {icon}
        <span className="text-[11px] font-black uppercase tracking-[0.16em]">{label}</span>
      </div>
      <p className="mt-2 text-sm font-semibold text-foreground">{value}</p>
    </div>
  )
}

function MessageBubble({
  message,
  onCopy,
  onAction,
}: {
  message: Message
  onCopy: (value: string, successMessage: string) => void
  onAction: (mode: 'simpler' | 'steps', content: string) => void
}) {
  const isUser = message.role === 'user'

  return (
    <div className={cn('flex', isUser ? 'justify-end' : 'justify-start')}>
      <div className={cn('group flex max-w-[88%] items-end gap-3', isUser && 'flex-row-reverse')}>
        <div
          className={cn(
            'flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl shadow-sm',
            isUser
              ? 'bg-gradient-to-br from-brand-500 to-brand-700 text-white'
              : 'bg-brand-500/12 text-brand-600 dark:bg-brand-500/20 dark:text-brand-300'
          )}
        >
          {isUser ? <span className="text-xs font-bold">You</span> : <Bot size={17} />}
        </div>

        <div className={cn('space-y-2', isUser && 'items-end')}>
          <div
            className={cn(
              'rounded-[24px] px-4 py-3 text-sm leading-7 shadow-sm',
              isUser
                ? 'rounded-br-md bg-gradient-to-r from-brand-500 to-brand-600 text-white shadow-brand-500/25'
                : 'rounded-bl-md border border-border/80 bg-surface text-foreground/85 dark:border-border dark:bg-slate-900 dark:text-slate-100'
            )}
          >
            <div className="whitespace-pre-wrap">
              {isUser ? message.content : renderRichText(message.content)}
            </div>
          </div>
          <div className={cn('flex items-center gap-2 px-1', isUser && 'justify-end')}>
            <p className="text-[11px] font-medium text-foreground/60">
              {isUser ? 'You' : 'MLForge AI'} • {formatTime(message.ts)}
            </p>

            {!isUser && (
              <div className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                <ActionChip
                  icon={<Copy size={12} />}
                  label="Copy"
                  onClick={() => onCopy(message.content, 'Message copied')}
                />
                <ActionChip
                  icon={<Wand2 size={12} />}
                  label="Simpler"
                  onClick={() => onAction('simpler', message.content)}
                />
                <ActionChip
                  icon={<ListChecks size={12} />}
                  label="Steps"
                  onClick={() => onAction('steps', message.content)}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

function InsightCard({
  title,
  description,
  icon,
  content,
  ok,
  disabled,
  disabledReason,
  loading,
  onClick,
  onUse,
}: {
  title: string
  description: string
  icon: ReactNode
  content: string
  ok: boolean | null
  disabled: boolean
  disabledReason: string | null
  loading: boolean
  onClick: () => void
  onUse: () => void
}) {
  const statusLabel = loading ? 'Analyzing' : ok === true ? 'Generated' : ok === false ? 'Failed' : 'Ready'
  const statusClass = loading
    ? 'border border-slate-200 bg-slate-100 text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300'
    : ok === true
      ? 'border border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-900/60 dark:bg-emerald-950/40 dark:text-emerald-300'
      : ok === false
        ? 'border border-rose-200 bg-rose-50 text-rose-700 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-300'
        : 'border border-brand-200 bg-brand-50 text-brand-700 dark:border-brand-900/60 dark:bg-brand-950/40 dark:text-brand-300'

  return (
    <div className="group relative rounded-3xl border border-border/70 bg-surface/90 px-4 py-4 text-left shadow-sm transition-all">
      <button
        type="button"
        onClick={onClick}
        disabled={disabled || loading}
        className={cn(
          'w-full text-left transition-all',
          !disabled && !loading && 'hover:-translate-y-0.5',
          disabled && 'cursor-not-allowed opacity-60'
        )}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex min-w-0 gap-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl bg-brand-500/10 text-brand-600 dark:bg-brand-500/20 dark:text-brand-300">
              {icon}
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-semibold text-foreground">{title}</h4>
                {!content && !loading && !disabled && (
                  <ChevronRight size={15} className="text-brand-500" />
                )}
              </div>
              <p className="mt-1 text-xs leading-5 text-foreground/70">
                {content ? 'Insight ready below.' : description}
              </p>
            </div>
          </div>

          <span className={cn('inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-bold', statusClass)}>
            {loading && <Loader2 size={13} className="animate-spin" />}
            {statusLabel}
          </span>
        </div>
      </button>

      {content && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          className="mt-4 border-t border-border/60 pt-4"
        >
          <div className="space-y-3">
            <div className="text-xs leading-6 text-foreground/75 whitespace-pre-wrap">
              {renderRichText(content)}
            </div>
            <div className="flex gap-2">
              <ActionChip
                icon={<Send size={12} />}
                label="Use in chat"
                onClick={onUse}
              />
            </div>
          </div>
        </motion.div>
      )}

      {disabled && disabledReason && (
        <div className="pointer-events-none absolute left-4 right-4 top-full z-10 mt-2 translate-y-1 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-xs font-medium text-amber-800 opacity-0 shadow-lg transition-all duration-150 group-hover:translate-y-0 group-hover:opacity-100 dark:border-amber-900/60 dark:bg-amber-950/90 dark:text-amber-200">
          {disabledReason}
        </div>
      )}
    </div>
  )
}

function ActionChip({
  icon,
  label,
  onClick,
}: {
  icon: ReactNode
  label: string
  onClick: () => void
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex items-center gap-1.5 rounded-full border border-border/70 bg-surface px-2.5 py-1 text-[11px] font-semibold text-foreground/85 shadow-sm transition-all hover:border-brand-300 hover:text-brand-600"
    >
      {icon}
      {label}
    </button>
  )
}
