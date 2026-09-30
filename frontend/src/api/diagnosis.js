import { get, post } from './index'

// 错误诊断 Agent：可诊断任务列表 / checkpoint 现场证据 / LLM 根因分析 / 多轮追问
export function getDiagnosisTasks() { return get('/api/diagnosis/tasks') }
export function getEvidence(taskId) { return get(`/api/diagnosis/tasks/${taskId}/evidence`) }
export function analyzeTask(taskId, errorLog = '') {
  return post('/api/diagnosis/analyze', { task_id: taskId, error_log: errorLog })
}
// 多轮诊断追问：首轮不带 sessionId（服务端创建并回传），后续轮携带以复用上下文
// subjectType: task=入库任务 | query=检索问答；query 时用 traceId 定位主体
export function chatDiagnosis(taskId, question, sessionId = null, errorLog = '', subjectType = 'task', traceId = null) {
  return post('/api/diagnosis/chat', {
    task_id: taskId, question, session_id: sessionId, error_log: errorLog,
    subject_type: subjectType, trace_id: traceId,
  })
}

// 检索/问答错误轨迹：列表（级别/会话过滤） / 单轮多步轨迹 / LLM 根因分析
export function getQueryErrors(level = '', conversationId = '') {
  const p = new URLSearchParams()
  if (level) p.set('level', level)
  if (conversationId) p.set('conversation_id', conversationId)
  const qs = p.toString()
  return get(`/api/diagnosis/query-errors${qs ? `?${qs}` : ''}`)
}
export function getQueryTrace(traceId) { return get(`/api/diagnosis/query-traces/${traceId}`) }
export function analyzeQuery(traceId, errorLog = '') {
  return post('/api/diagnosis/analyze-query', { trace_id: traceId, error_log: errorLog })
}
