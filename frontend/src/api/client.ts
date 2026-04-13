import axios from 'axios'

const api = axios.create({ baseURL: '/', timeout: 30000 })


export const uploadFile = (file: File) => {
  const fd = new FormData()
  fd.append('file', file)
  return api.post('/api/data/upload', fd, { headers: { 'Content-Type': 'multipart/form-data' } })
}
export const previewData = (sessionId: string, rows = 50) =>
  api.get('/api/data/preview', { params: { session_id: sessionId, rows } })

export const getSessions = () => api.get('/api/data/sessions')
export const getSession = (id: string) => api.get(`/api/data/sessions/${id}`)
export const getSamples = () => api.get('/api/data/samples')
export const loadSample = (name: string) => api.post('/api/data/load-sample', null, { params: { name } })


export const applyStep = (sessionId: string, stepType: string, params = {}) =>
  api.post('/api/preprocess/apply', { session_id: sessionId, step_type: stepType, params })

export const undoStep = (sessionId: string) =>
  api.post('/api/preprocess/undo', null, { params: { session_id: sessionId } })

export const resetData = (sessionId: string) =>
  api.post('/api/preprocess/reset', null, { params: { session_id: sessionId } })

export const getSteps = (sessionId: string) =>
  api.get('/api/preprocess/steps', { params: { session_id: sessionId } })

export const selectFeatures = (sessionId: string, featureCols: string[], targetCol: string, taskType: string) =>
  api.post('/api/preprocess/select-features', null, {
    params: { session_id: sessionId, target_col: targetCol, task_type: taskType },
    data: featureCols,
  })

export const selectFeaturesBody = (sessionId: string, featureCols: string[], targetCol: string, taskType: string) =>
  api.post('/api/preprocess/select-features', featureCols, {
    params: { session_id: sessionId, target_col: targetCol, task_type: taskType },
  })

export const suggestFeatures = (sessionId: string) =>
  api.get('/api/preprocess/suggest-features', { params: { session_id: sessionId } })


export const getAvailableModels = (taskType?: string) =>
  api.get('/api/train/models', { params: taskType ? { task_type: taskType } : {} })

export const startTraining = (payload: {
  session_id: string; model_name: string; model_type?: string
  hyperparams?: object; cv_folds?: number; test_size?: number
  feature_cols?: string[]; target_col?: string; task_type?: string
}) => api.post('/api/train/start', payload)

export const getTrainStatus = (runId: string) => api.get(`/api/train/status/${runId}`)
export const cancelRun = (runId: string) => api.post(`/api/train/cancel/${runId}`)
export const getRuns = (sessionId: string) => api.get('/api/train/runs', { params: { session_id: sessionId } })


export const getMetrics = (runId: string) => api.get(`/api/eval/metrics/${runId}`)
export const compareModels = (sessionId: string, runIds: string[]) =>
  api.post('/api/eval/compare', { session_id: sessionId, run_ids: runIds })
export const getFeatureImportance = (runId: string) => api.get(`/api/eval/feature-importance/${runId}`)


export const predictSingle = (sessionId: string, runId: string, inputs: Record<string, number>) =>
  api.post('/api/predict/single', { session_id: sessionId, run_id: runId, inputs })

export const predictBatch = (runId: string, sessionId: string, file: File) => {
  const fd = new FormData()
  fd.append('run_id', runId)
  fd.append('session_id', sessionId)
  fd.append('file', file)
  return api.post('/api/predict/batch', fd, { responseType: 'blob' })
}


export const exportModel = (runId: string) =>
  api.get(`/api/export/model/${runId}`, { responseType: 'blob' })

export const exportPowerBI = (runId: string) =>
  api.get(`/api/export/powerbi/${runId}`, { responseType: 'blob' })

export default api
