import { create } from 'zustand'
import { createJSONStorage, persist } from 'zustand/middleware'


export interface SessionInfo {
  session_id: string
  filename: string
  dataset_id?: string
  n_rows: number
  n_cols: number
  column_names: string[]
  feature_cols: string[]
  target_col: string | null
  task_type: string | null
}

export interface RunInfo {
  run_id: string
  model_name: string
  status: 'queued' | 'running' | 'complete' | 'failed' | 'cancelled'
  progress: number
  task_type: string
  training_time_s: number | null
  primary_metric: number | null
  started_at: string
}

export interface MetricsData {
  run_id: string
  model_name: string
  task_type: string
  metrics: Record<string, number | number[][] | string[]>
  cv_scores: number[]
  feature_importance: Record<string, number>
  confusion_matrix: number[][] | null
  roc_data: { fpr: number[]; tpr: number[]; auc: number } | null
  pr_data: { precision: number[]; recall: number[] } | null
  classes: string[] | null
  training_time_s: number | null
  feature_cols: string[]
  target_col: string
  params: Record<string, unknown>
}


interface SessionStore {
  session: SessionInfo | null
  preview: Record<string, string>[]
  steps: Array<{ step_type: string; params: object }>
  setSession: (s: SessionInfo) => void
  setPreview: (p: Record<string, string>[]) => void
  setSteps: (s: SessionStore['steps']) => void
  clearSession: () => void
  setFeatureSelection: (featureCols: string[], targetCol: string, taskType: string) => void
  updatePreviewFromApply: (result: { preview_head: Record<string, string>[]; column_names: string[] }) => void
}

export const useSessionStore = create<SessionStore>()(
  persist(
    (set) => ({
      session: null,
      preview: [],
      steps: [],
      setSession: (s) => set({ session: s }),
      setPreview: (p) => set({ preview: p }),
      setSteps: (s) => set({ steps: s }),
      clearSession: () => set({ session: null, preview: [], steps: [] }),
      setFeatureSelection: (featureCols, targetCol, taskType) =>
        set((state) => ({
          session: state.session
            ? { ...state.session, feature_cols: featureCols, target_col: targetCol, task_type: taskType }
            : null,
        })),
      updatePreviewFromApply: (result) =>
        set((state) => ({
          preview: result.preview_head,
          session: state.session ? { ...state.session, column_names: result.column_names } : null,
        })),
    }),
    {
      name: 'ml-platform-session',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        session: state.session,
        steps: state.steps,
      }),
    }
  )
)


interface ModelStore {
  runs: RunInfo[]
  activeRunId: string | null
  metrics: Record<string, MetricsData>
  setRuns: (r: RunInfo[]) => void
  addRun: (r: RunInfo) => void
  updateRun: (runId: string, updates: Partial<RunInfo>) => void
  setActiveRun: (id: string | null) => void
  setMetrics: (runId: string, m: MetricsData) => void
}

export const useModelStore = create<ModelStore>()(
  persist(
    (set) => ({
      runs: [],
      activeRunId: null,
      metrics: {},
      setRuns: (r) => set({ runs: r }),
      addRun: (r) => set((state) => ({ runs: [r, ...state.runs] })),
      updateRun: (runId, updates) =>
        set((state) => ({
          runs: state.runs.map((r) => (r.run_id === runId ? { ...r, ...updates } : r)),
        })),
      setActiveRun: (id) => set({ activeRunId: id }),
      setMetrics: (runId, m) => set((state) => ({ metrics: { ...state.metrics, [runId]: m } })),
    }),
    {
      name: 'ml-platform-models',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        runs: state.runs,
        activeRunId: state.activeRunId,
      }),
    }
  )
)


interface UIStore {
  sidebarOpen: boolean
  darkMode: boolean
  isAiDrawerOpen: boolean
  notification: { type: 'success' | 'error' | 'info'; message: string } | null
  setSidebarOpen: (v: boolean) => void
  toggleDarkMode: () => void
  setDarkMode: (v: boolean) => void
  toggleAiDrawer: () => void
  setAiDrawerOpen: (v: boolean) => void
  notify: (type: UIStore['notification'] extends null ? never : UIStore['notification']['type'], message: string) => void
  clearNotification: () => void
}

export const useUIStore = create<UIStore>()(
  persist(
    (set) => ({
      sidebarOpen: true,
      darkMode: true, // Default to dark mode for premium feel
      isAiDrawerOpen: false,
      notification: null,
      setSidebarOpen: (v) => set({ sidebarOpen: v }),
      toggleDarkMode: () => set((state) => ({ darkMode: !state.darkMode })),
      setDarkMode: (v) => set({ darkMode: v }),
      toggleAiDrawer: () => set((state) => ({ isAiDrawerOpen: !state.isAiDrawerOpen })),
      setAiDrawerOpen: (v) => set({ isAiDrawerOpen: v }),
      notify: (type, message) => set({ notification: { type, message } }),
      clearNotification: () => set({ notification: null }),
    }),
    {
      name: 'ml-platform-ui',
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        darkMode: state.darkMode,
      }),
    }
  )
)
