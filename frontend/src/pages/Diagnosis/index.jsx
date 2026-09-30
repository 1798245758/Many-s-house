import { useState, useEffect, useCallback, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { getDiagnosisTasks, getEvidence, chatDiagnosis, getQueryErrors, getQueryTrace, analyzeQuery } from '../../api/diagnosis'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

const DOC_STATUS_LABELS = { error: '失败', cancelled: '已取消' }

const LEVEL_TAG = { error: 'tag tag-danger', degraded: 'tag tag-warn', normal: 'tag' }
const LEVEL_LABEL = { error: '错误', degraded: '降级', normal: '正常' }

const SUGGESTIONS = [
  '这个任务失败的根本原因是什么？',
  '从事件日志看，失败发生在哪个阶段？',
  '我需要怎么修复才能重新入库？',
  '还需要我提供哪些额外信息来进一步定位？',
]

function EvidencePanel({ evidence }) {
  if (!evidence) return null
  return (
    <div className="card">
      <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>🔎 CheckPoint 现场证据</h3>
      <div className="row" style={{ gap: 8, margin: '12px 0' }}>
        <span className="tag">任务 {evidence.task_id.slice(0, 8)}…</span>
        <span className="tag tag-danger">状态 {evidence.final_status}</span>
        {evidence.final_stage && <span className="tag tag-warn">阶段 {evidence.final_stage}</span>}
        {evidence.retry_count > 0 && <span className="tag">重试 {evidence.retry_count} 次</span>}
      </div>
      {evidence.error && (
        <div className="error-message" style={{ animation: 'none', wordBreak: 'break-all' }}>{evidence.error}</div>
      )}
      {evidence.document && (
        <p className="muted" style={{ fontSize: '0.85rem', marginBottom: 10 }}>
          📄 {evidence.document.filename}（{evidence.document.file_type} · {(evidence.document.file_size / 1024).toFixed(1)}KB · {DOC_STATUS_LABELS[evidence.document.status] || evidence.document.status}）
        </p>
      )}
      <details>
        <summary>状态机流转事件日志（{evidence.events.length} 条）</summary>
        <pre className="code-block" style={{ maxHeight: 260 }}>
          {evidence.events.map((e, i) =>
            `#${i + 1} [${e.status}]${e.stage ? ` stage=${e.stage}` : ''}${e.progress != null ? ` ${e.progress}%` : ''} ${e.message || ''}${e.error ? ` | 错误: ${e.error}` : ''}`
          ).join('\n')}
        </pre>
      </details>
    </div>
  )
}

// 上下文管理仪表盘：直观展示「多轮追问但上下文有界、证据只钉一次、旧轮折叠进摘要」
function ContextMeter({ stats }) {
  if (!stats) return null
  const pct = Math.min(100, Math.round((stats.context_chars / stats.budget) * 100))
  return (
    <div className="ctx-stats">
      <span className="ctx-item">🧠 上下文管理</span>
      <span className="ctx-item">累计轮次 <b>{stats.total_turns}</b></span>
      <span className="ctx-item">窗口保留 <b>{stats.window_turns}</b></span>
      <span className="ctx-item">证据钉住 <b>{(stats.evidence_chars / 1000).toFixed(1)}k</b>（仅一次）</span>
      <span className="ctx-item">滚动摘要 <b>{stats.summary_chars}</b> 字</span>
      <span className="ctx-item">压缩触发 <b>{stats.compact_count}</b> 次</span>
      <span className="ctx-item" title={`上下文 ${stats.context_chars} / 预算 ${stats.budget} 字符`}>
        占用 <b>{pct}%</b>
        <span className="ctx-meter"><i style={{ width: `${pct}%` }} /></span>
      </span>
    </div>
  )
}

export default function Diagnosis() {
  const [tab, setTab] = useState('task')
  return (
    <div style={{ maxWidth: 900, margin: '0 auto' }}>
      <h1 className="page-title">错误诊断 Agent</h1>
      <p className="page-subtitle">
        基于 LangGraph CheckPoint 的状态机流转与关键信息，结合错误日志，由大模型分析失败根因，支持多轮追问（证据钉住 + 滑动窗口 + 滚动摘要，上下文有界不混乱）。
      </p>
      <div className="row" style={{ gap: 8, marginBottom: 14 }}>
        <button className={`btn btn-sm ${tab === 'task' ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setTab('task')}>📦 入库任务</button>
        <button className={`btn btn-sm ${tab === 'query' ? 'btn-primary' : 'btn-ghost'}`} onClick={() => setTab('query')}>🔍 检索问答</button>
      </div>
      {tab === 'task' ? <TaskPanel /> : <QueryPanel />}
    </div>
  )
}

// ===== 检索/问答错误标签：每轮多步轨迹列表 → 展开轨迹 → LLM 根因分析 → 多轮追问 =====
const QUERY_SUGGESTIONS = [
  '这轮检索为什么降级/失败？',
  '候选集为空是哪个环节造成的？',
  '证据不足时我该怎么补充知识库？',
  '还需要我提供哪些信息来进一步定位？',
]

function TraceSteps({ steps }) {
  if (!steps || steps.length === 0) return <p className="muted" style={{ fontSize: '0.85rem' }}>无多步轨迹</p>
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6, margin: '10px 0' }}>
      {steps.map((s, i) => (
        <div key={i} className="row" style={{ gap: 8, alignItems: 'flex-start', padding: '6px 8px', borderRadius: 8,
          background: s.level === 'error' ? 'rgba(239,68,68,0.08)' : s.level === 'degraded' ? 'rgba(245,158,11,0.08)' : 'transparent' }}>
          <span className="muted" style={{ fontSize: '0.8rem', minWidth: 18 }}>{i + 1}</span>
          <span className={LEVEL_TAG[s.level] || 'tag'}>{s.step}</span>
          <span style={{ flex: 1, fontSize: '0.88rem' }}>
            {s.code && <b className="mono">{s.code}</b>} {s.message || ''}
            {s.metrics && Object.keys(s.metrics).length > 0 && (
              <span className="muted mono" style={{ fontSize: '0.75rem' }}> {JSON.stringify(s.metrics)}</span>
            )}
          </span>
        </div>
      ))}
    </div>
  )
}

function QueryPanel() {
  const [traces, setTraces] = useState([])
  const [level, setLevel] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [traceId, setTraceId] = useState(null)
  const [detail, setDetail] = useState(null)   // collect_query_evidence
  const [analysis, setAnalysis] = useState('')
  const [analyzing, setAnalyzing] = useState(false)
  const [errorLog, setErrorLog] = useState('')

  // 多轮追问会话（与任务侧同构，subject_type='query'）
  const [sessionId, setSessionId] = useState(null)
  const [turns, setTurns] = useState([])
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [lastStats, setLastStats] = useState(null)
  const bottomRef = useRef(null)

  const loadList = useCallback(async (lv) => {
    setLoading(true); setError('')
    try { setTraces(await getQueryErrors(lv)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { loadList(level) }, [level, loadList])
  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [turns, thinking])

  const openTrace = useCallback(async (id) => {
    setError(''); setDetail(null); setAnalysis(''); setTurns([]); setSessionId(null); setLastStats(null); setTraceId(id)
    try { setDetail(await getQueryTrace(id)) }
    catch (e) { setError(e.message) }
  }, [])

  const runAnalyze = async () => {
    if (!traceId || analyzing) return
    setAnalyzing(true); setError('')
    try { const r = await analyzeQuery(traceId, errorLog); setAnalysis(r.analysis || '') }
    catch (e) { setError(e.message) }
    finally { setAnalyzing(false) }
  }

  const ask = async (text) => {
    const q = (text ?? input).trim()
    if (!q || thinking || !traceId) return
    setInput('')
    setTurns(t => [...t, { role: 'user', text: q }])
    setThinking(true)
    try {
      const data = await chatDiagnosis(null, q, sessionId, sessionId ? '' : errorLog, 'query', traceId)
      setSessionId(data.session_id)
      setLastStats(data.context_stats)
      setTurns(t => [...t, { role: 'bot', text: data.answer, stats: data.context_stats }])
    } catch (e) {
      setError(e.message)
      setTurns(t => [...t, { role: 'bot', text: '', llm_error: e.message }])
    } finally { setThinking(false) }
  }

  const resetSession = () => { setTurns([]); setSessionId(null); setLastStats(null) }

  return (
    <>
      {error && <ErrorMessage message={error} />}
      <div className="card">
        <div className="row-between" style={{ alignItems: 'center' }}>
          <label>检索/问答轨迹（每轮多步状态机，出错步骤高亮）</label>
          <select value={level} onChange={e => setLevel(e.target.value)} style={{ width: 'auto' }}>
            <option value="">全部级别</option>
            <option value="error">错误</option>
            <option value="degraded">降级</option>
            <option value="normal">正常</option>
          </select>
        </div>
        {loading ? <Loading text="加载轨迹" /> : traces.length === 0 ? (
          <p className="muted" style={{ marginTop: 8, fontSize: '0.85rem' }}>暂无检索轨迹</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginTop: 10 }}>
            {traces.map(t => (
              <div key={t.trace_id} onClick={() => openTrace(t.trace_id)}
                className="row" style={{ gap: 8, alignItems: 'center', cursor: 'pointer', padding: '8px 10px', borderRadius: 8,
                  background: traceId === t.trace_id ? 'rgba(255,255,255,0.06)' : 'transparent' }}>
                <span className={LEVEL_TAG[t.final_level] || 'tag'}>{LEVEL_LABEL[t.final_level] || t.final_level}</span>
                {t.final_code && <b className="mono" style={{ fontSize: '0.82rem' }}>{t.final_code}</b>}
                <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.9rem' }}>{t.question}</span>
                <span className="muted" style={{ fontSize: '0.75rem', whiteSpace: 'nowrap' }}>{t.created_at}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {detail && (
        <>
          <div className="card">
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>🔎 检索现场证据</h3>
            <div className="row" style={{ gap: 8, margin: '12px 0', flexWrap: 'wrap' }}>
              <span className="tag">轨迹 #{detail.trace_id}</span>
              <span className={LEVEL_TAG[detail.final_level] || 'tag'}>{LEVEL_LABEL[detail.final_level] || detail.final_level} {detail.final_code}</span>
              {detail.response_type && <span className="tag">类型 {detail.response_type}</span>}
              {detail.conversation_id && <span className="tag">会话 {detail.conversation_id.slice(0, 8)}</span>}
            </div>
            <p style={{ fontSize: '0.9rem', margin: '4px 0' }}><b>问：</b>{detail.question}</p>
            {detail.answer_snippet && <p className="muted" style={{ fontSize: '0.85rem', margin: '4px 0' }}><b>答：</b>{detail.answer_snippet}</p>}
            <TraceSteps steps={detail.steps} />
          </div>

          {turns.length === 0 && !analysis && (
            <div className="card">
              <label>附加错误日志（可选，分析/首轮提问时用于采集证据）</label>
              <textarea value={errorLog} onChange={e => setErrorLog(e.target.value)} rows={3}
                placeholder="Traceback ..." className="mono" style={{ fontSize: '0.8rem' }} />
              <div className="row" style={{ marginTop: 10 }}>
                <button className="btn btn-primary" onClick={runAnalyze} disabled={analyzing}>
                  {analyzing ? '分析中…' : '🩺 分析根因'}
                </button>
              </div>
            </div>
          )}

          {analysis && (
            <div className="card">
              <div className="row-between" style={{ alignItems: 'center' }}>
                <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>🧠 根因分析</h3>
                <button className="btn btn-sm btn-ghost" onClick={() => setAnalysis('')}>✕ 收起</button>
              </div>
              <div className="answer-text">{analysis}</div>
            </div>
          )}

          <div className="glass-panel" style={{ padding: 18, minHeight: 220, marginBottom: 14 }}>
            <div className="row-between" style={{ marginBottom: 14 }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>🩺 诊断对话</h3>
              {turns.length > 0 && <button className="btn btn-sm btn-ghost" onClick={resetSession}>↺ 新会话</button>}
            </div>

            {turns.length === 0 && !thinking && (
              <div className="empty-state">
                <div className="icon">💡</div>
                <p>向诊断 Agent 提问这轮检索，可连续追问（上下文自动管理）</p>
                <div className="row" style={{ justifyContent: 'center', marginTop: 16, flexWrap: 'wrap' }}>
                  {QUERY_SUGGESTIONS.map(s => (
                    <span key={s} className="tag" style={{ cursor: 'pointer' }} onClick={() => ask(s)}>{s}</span>
                  ))}
                </div>
              </div>
            )}

            {turns.map((t, i) => t.role === 'user'
              ? <div key={i} className="chat-row"><div className="chat-bubble chat-user">{t.text}</div></div>
              : (
                <div key={i} className="chat-row">
                  <div className="chat-bubble chat-bot" style={{ width: '100%', maxWidth: '88%' }}>
                    {t.llm_error
                      ? <span style={{ color: 'var(--danger)' }}>⚠️ {t.llm_error}</span>
                      : <div className="answer-text">{t.text}</div>}
                    {t.stats && <ContextMeter stats={t.stats} />}
                  </div>
                </div>
              ))}

            {thinking && (
              <div className="chat-row">
                <div className="chat-bubble chat-bot typing"><span></span><span></span><span></span></div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <form className="glass-panel" style={{ padding: 14, display: 'flex', gap: 10, alignItems: 'flex-end' }}
            onSubmit={e => { e.preventDefault(); ask() }}>
            <textarea rows={2} placeholder="继续追问，例如：候选集为空该怎么排查？（Enter 发送 / Shift+Enter 换行）"
              value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask() } }}
              style={{ flex: 1 }} />
            <button className="btn btn-primary" type="submit" disabled={thinking || !input.trim()}>
              {thinking ? '分析中' : '发送 ↑'}
            </button>
          </form>

          {lastStats && lastStats.compact_count > 0 && (
            <p className="muted" style={{ fontSize: '0.8rem', marginTop: 12, textAlign: 'center' }}>
              ✨ 已触发 {lastStats.compact_count} 次上下文压缩：早期轮次折叠为滚动摘要，现场证据始终钉住，多轮追问不丢失关键信息。
            </p>
          )}
        </>
      )}
    </>
  )
}

function TaskPanel() {
  const [searchParams] = useSearchParams()
  const [tasks, setTasks] = useState([])
  const [taskId, setTaskId] = useState(searchParams.get('task_id') || '')
  const [evidence, setEvidence] = useState(null)
  const [errorLog, setErrorLog] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // 多轮追问会话
  const [sessionId, setSessionId] = useState(null)
  const [turns, setTurns] = useState([])   // {role:'user'|'bot', text, stats, llm_error}
  const [input, setInput] = useState('')
  const [thinking, setThinking] = useState(false)
  const [lastStats, setLastStats] = useState(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    getDiagnosisTasks().then(setTasks).catch(e => setError(e.message)).finally(() => setLoading(false))
  }, [])

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [turns, thinking])

  // 切换任务：加载证据并重置会话（跨任务不共享上下文，避免污染）
  const loadEvidence = useCallback(async (id) => {
    if (!id) return
    setError(''); setEvidence(null); setTurns([]); setSessionId(null); setLastStats(null)
    try { setEvidence(await getEvidence(id)) }
    catch (e) { setError(e.message) }
  }, [])

  useEffect(() => { if (taskId) loadEvidence(taskId) }, [taskId, loadEvidence])

  const ask = async (text) => {
    const q = (text ?? input).trim()
    if (!q || thinking || !taskId) return
    setInput('')
    setTurns(t => [...t, { role: 'user', text: q }])
    setThinking(true)
    try {
      // 首轮携带错误日志用于采集证据；后续轮复用会话（sessionId）
      const data = await chatDiagnosis(taskId, q, sessionId, sessionId ? '' : errorLog)
      setSessionId(data.session_id)
      setLastStats(data.context_stats)
      setTurns(t => [...t, { role: 'bot', text: data.answer, stats: data.context_stats }])
    } catch (e) {
      setError(e.message)
      setTurns(t => [...t, { role: 'bot', text: '', llm_error: e.message }])
    } finally {
      setThinking(false)
    }
  }

  const resetSession = () => { setTurns([]); setSessionId(null); setLastStats(null) }

  return (
    <>
      {error && <ErrorMessage message={error} />}

      <div className="card">
        <label>选择失败任务</label>
        <select value={taskId} onChange={e => setTaskId(e.target.value)}>
          <option value="">— 请选择任务 —</option>
          {tasks.map(t => (
            <option key={t.task_id} value={t.task_id}>
              {t.filename}（{DOC_STATUS_LABELS[t.doc_status] || t.doc_status} · {t.task_id.slice(0, 8)}…）
            </option>
          ))}
        </select>
        {!loading && tasks.length === 0 && <p className="muted" style={{ marginTop: 8, fontSize: '0.85rem' }}>暂无可诊断的失败任务</p>}
      </div>

      {loading ? <Loading text="加载任务" /> : evidence && (
        <>
          <EvidencePanel evidence={evidence} />

          {turns.length === 0 && (
            <div className="card">
              <label>附加错误日志（可选，首轮提问时用于采集证据）</label>
              <textarea value={errorLog} onChange={e => setErrorLog(e.target.value)} rows={3}
                placeholder="Traceback ..." className="mono" style={{ fontSize: '0.8rem' }} />
            </div>
          )}

          {/* 多轮诊断对话区 */}
          <div className="glass-panel" style={{ padding: 18, minHeight: 260, marginBottom: 14 }}>
            <div className="row-between" style={{ marginBottom: 14 }}>
              <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>🩺 诊断对话</h3>
              {turns.length > 0 && <button className="btn btn-sm btn-ghost" onClick={resetSession}>↺ 新会话</button>}
            </div>

            {turns.length === 0 && !thinking && (
              <div className="empty-state">
                <div className="icon">💡</div>
                <p>向诊断 Agent 提问，可连续追问（上下文自动管理）</p>
                <div className="row" style={{ justifyContent: 'center', marginTop: 16 }}>
                  {SUGGESTIONS.map(s => (
                    <span key={s} className="tag" style={{ cursor: 'pointer' }} onClick={() => ask(s)}>{s}</span>
                  ))}
                </div>
              </div>
            )}

            {turns.map((t, i) => t.role === 'user'
              ? <div key={i} className="chat-row"><div className="chat-bubble chat-user">{t.text}</div></div>
              : (
                <div key={i} className="chat-row">
                  <div className="chat-bubble chat-bot" style={{ width: '100%', maxWidth: '88%' }}>
                    {t.llm_error
                      ? <span style={{ color: 'var(--danger)' }}>⚠️ {t.llm_error}</span>
                      : <div className="answer-text">{t.text}</div>}
                    {t.stats && <ContextMeter stats={t.stats} />}
                  </div>
                </div>
              ))}

            {thinking && (
              <div className="chat-row">
                <div className="chat-bubble chat-bot typing"><span></span><span></span><span></span></div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <form className="glass-panel" style={{ padding: 14, display: 'flex', gap: 10, alignItems: 'flex-end' }}
            onSubmit={e => { e.preventDefault(); ask() }}>
            <textarea rows={2} placeholder="继续追问，例如：那具体该怎么修复？（Enter 发送 / Shift+Enter 换行）"
              value={input} onChange={e => setInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask() } }}
              style={{ flex: 1 }} />
            <button className="btn btn-primary" type="submit" disabled={thinking || !input.trim()}>
              {thinking ? '分析中' : '发送 ↑'}
            </button>
          </form>

          {lastStats && lastStats.compact_count > 0 && (
            <p className="muted" style={{ fontSize: '0.8rem', marginTop: 12, textAlign: 'center' }}>
              ✨ 已触发 {lastStats.compact_count} 次上下文压缩：早期轮次折叠为滚动摘要，现场证据始终钉住，多轮追问不丢失关键信息。
            </p>
          )}
        </>
      )}
    </>
  )
}
