import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { useSessionStore, useModelStore, useUIStore } from '../store'
import api from '../api/client'
import { Bot, X, Send, Sparkles, Brain, Activity, ChevronRight } from 'lucide-react'
import { cn } from '../lib/utils'

interface Message { role: 'user' | 'assistant'; content: string; ts: number }

const STARTERS = [
  "What preprocessing steps should I apply to this dataset?",
  "Which model should I use for classification?",
  "How do I interpret my model's accuracy?",
  "What does feature importance mean?",
]

export default function AIAssistantDrawer() {
  const session = useSessionStore(s => s.session)
  const runs = useModelStore(s => s.runs)
  const activeRunId = useModelStore(s => s.activeRunId)
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
  const [schemaInsight, setSchemaInsight] = useState('')
  const [modelRec, setModelRec] = useState('')
  const [metricsInsight, setMetricsInsight] = useState('')
  const [loadingInsight, setLoadingInsight] = useState('')
  const bottomRef = useRef<HTMLDivElement>(null)

  const completedRuns = runs.filter(r => r.status === 'complete')
  const currentRunId = activeRunId || completedRuns[0]?.run_id

  useEffect(() => {
    if (isAiDrawerOpen) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages, isAiDrawerOpen])

  const send = async (text: string) => {
    if (!text.trim() || loading) return
    const userMsg: Message = { role: 'user', content: text, ts: Date.now() }
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
      setMessages(m => [...m, { role: 'assistant', content: res.data.response, ts: Date.now() }])
    } catch { notify('error', 'AI chat unavailable — check GEMINI_API_KEY') } finally { setLoading(false) }
  }

  const loadInsight = async (type: 'schema' | 'model' | 'metrics') => {
    setLoadingInsight(type)
    try {
      if (type === 'schema') {
        const r = await api.get(`/api/ai/schema/${session?.session_id}`)
        setSchemaInsight(r.data.narrative)
      } else if (type === 'model') {
        const r = await api.get(`/api/ai/recommend-model/${session?.session_id}`, {
          params: { task_type: session?.task_type || 'classification', target_col: session?.target_col }
        })
        setModelRec(r.data.recommendation)
      } else if (type === 'metrics') {
        const r = await api.get(`/api/ai/interpret-metrics/${currentRunId}`)
        setMetricsInsight(r.data.interpretation)
      }
    } catch {
      notify('error', `Failed to load ${type} insight`)
    } finally { setLoadingInsight('') }
  }

  return (
    <AnimatePresence>
      {isAiDrawerOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setAiDrawerOpen(false)}
            className="fixed inset-0 bg-slate-900/25 z-40 transition-opacity"
          />
          <motion.div
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed right-0 top-0 bottom-0 w-full sm:w-[min(92vw,560px)] lg:w-[620px] bg-surface/98 backdrop-blur-sm border-l border-border shadow-2xl z-50 flex flex-col overflow-hidden"
          >
            {/* Header */}
            <div className="flex items-center justify-between px-4 sm:px-6 py-4 border-b border-border bg-surface-hover/30">
              <div className="flex items-center gap-3">
                <div className="p-2 bg-brand-500/10 rounded-xl text-brand-500">
                  <Bot size={22} className="text-brand-500" />
                </div>
                <div>
                  <h2 className="text-lg font-bold text-foreground leading-tight">AI Assistant</h2>
                  <p className="text-xs text-brand-500 font-medium tracking-wide uppercase">Context-Aware ML Guide</p>
                </div>
              </div>
              <button
                onClick={() => setAiDrawerOpen(false)}
                className="p-2 rounded-full text-slate-400 hover:text-foreground hover:bg-surface-hover transition-colors"
              >
                <X size={20} />
              </button>
            </div>

            {/* Content Area */}
            <div className="flex-1 overflow-y-auto w-full min-w-0 custom-scrollbar">

              {/* Insight Cards Area (Expandable/Compact) */}
              <div className="px-4 sm:px-5 pt-5 pb-2 flex flex-col gap-3">
                <InsightCard
                  title="Dataset Insight"
                  icon={<Sparkles size={16} />}
                  content={schemaInsight}
                  loading={loadingInsight === 'schema'}
                  onClick={() => loadInsight('schema')}
                  disabled={!session || !!schemaInsight}
                />
                <InsightCard
                  title="Model Recommendation"
                  icon={<Brain size={16} />}
                  content={modelRec}
                  loading={loadingInsight === 'model'}
                  onClick={() => loadInsight('model')}
                  disabled={!session?.target_col || !!modelRec}
                />
                <InsightCard
                  title="Metrics Interpretation"
                  icon={<Activity size={16} />}
                  content={metricsInsight}
                  loading={loadingInsight === 'metrics'}
                  onClick={() => loadInsight('metrics')}
                  disabled={!currentRunId || !!metricsInsight}
                />
              </div>

              {/* Chat Messages */}
              <div className="px-4 sm:px-5 py-4 flex flex-col gap-4">
                {messages.map((msg, i) => (
                  <div key={i} className={cn("flex", msg.role === 'user' ? "justify-end" : "justify-start")}>
                    <div className={cn(
                      "max-w-[90%] px-4 py-3 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap",
                      msg.role === 'user'
                        ? "bg-brand-600 text-white rounded-br-sm shadow-md shadow-brand-600/20"
                        : "bg-surface-hover text-foreground rounded-bl-sm border border-border"
                    )}>
                      {msg.content}
                    </div>
                  </div>
                ))}
                {loading && (
                  <div className="flex justify-start">
                    <div className="bg-surface-hover px-4 py-3 rounded-2xl rounded-bl-sm border border-border text-slate-500 text-sm flex gap-2 items-center">
                      <span className="animate-pulse">●</span>
                      <span className="animate-pulse animation-delay-200">●</span>
                      <span className="animate-pulse animation-delay-400">●</span>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} className="h-2" />
              </div>

            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-border bg-surface-hover/30">
              {messages.length < 3 && (
                <div className="mb-3 flex flex-col gap-2">
                  {STARTERS.map((s, i) => (
                    <button
                      key={i}
                      onClick={() => send(s)}
                      className="text-left text-xs text-foreground/80 hover:text-brand-500 font-medium py-2 px-3 bg-surface border border-border rounded-lg hover:border-brand-500/50 transition-all truncate"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}

              <div className="flex items-end gap-2 bg-surface border border-border focus-within:border-brand-500 focus-within:ring-1 focus-within:ring-brand-500/30 rounded-xl p-1 transition-all">
                <textarea
                  value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      send(input);
                    }
                  }}
                  placeholder="Ask about your data, models..."
                  className="flex-1 max-h-32 min-h-10 bg-transparent text-sm text-foreground placeholder:text-slate-500 outline-none resize-none px-3 py-2.5"
                  rows={Math.min(4, input.split('\n').length)}
                />
                <button
                  onClick={() => send(input)}
                  disabled={loading || !input.trim()}
                  className="p-2 mb-1 mr-1 rounded-lg bg-brand-500 hover:bg-brand-600 text-white disabled:opacity-50 disabled:hover:bg-brand-500 transition-colors"
                >
                  <Send size={18} />
                </button>
              </div>

              {(session || currentRunId) && (
                <div className="mt-3 flex flex-wrap gap-2 sm:gap-3 text-[10px] font-medium text-slate-500 uppercase tracking-wider px-1 overflow-hidden">
                  {session && <span className="truncate flex-shrink">📄 {session.filename}</span>}
                  {session?.target_col && <span className="truncate flex-shrink-0 text-brand-500">🎯 {session.target_col}</span>}
                  {currentRunId && <span className="truncate flex-shrink text-emerald-500">🤖 {runs.find(r => r.run_id === currentRunId)?.model_name}</span>}
                </div>
              )}
            </div>

          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}

function InsightCard({ title, icon, content, disabled, loading, onClick }: any) {
  return (
    <div className="border border-border bg-surface-hover/50 rounded-xl overflow-hidden transition-all hover:bg-surface-hover group">
      <button
        onClick={onClick}
        disabled={disabled || loading}
        className="w-full flex items-center justify-between px-4 py-3 text-left disabled:opacity-70 disabled:cursor-not-allowed"
      >
        <div className="flex items-center gap-2">
          <div className="text-brand-500">{icon}</div>
          <span className="font-semibold text-sm text-foreground">{title}</span>
        </div>
        {!content && !loading && (
          <div className="text-brand-500 opacity-0 group-hover:opacity-100 transition-opacity">
            <ChevronRight size={16} />
          </div>
        )}
        {loading && <span className="text-slate-400 text-xs font-medium">Analyzing...</span>}
        {content && !loading && <span className="text-emerald-500 font-bold text-xs">Generated</span>}
      </button>
      <AnimatePresence>
        {content && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            className="px-4 pb-4"
          >
            <div className="pt-3 border-t border-border/50 text-xs text-foreground/80 leading-relaxed">
              {content}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
