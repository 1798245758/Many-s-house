import { useState, useEffect, useRef } from 'react'
import { useSearchParams } from 'react-router-dom'
import { submitQuery } from '../../api/query'
import ErrorMessage from '../../components/ErrorMessage'

const CONV_ID_KEY = 'conversation_id'
function getConversationId() {
  let id = sessionStorage.getItem(CONV_ID_KEY)
  if (!id) { id = crypto.randomUUID(); sessionStorage.setItem(CONV_ID_KEY, id) }
  return id
}

// 各响应类型的呈现样式
const TYPE_META = {
  clarification: { label: '需要补充信息', cls: 'tag-warn', icon: '❓' },
  refusal: { label: '无法回答', cls: 'tag-danger', icon: '⛔' },
  permission_denied: { label: '权限不足', cls: 'tag-warn', icon: '🔒' },
  answer: { label: '回答', cls: 'tag-success', icon: '✅' },
}

function IntentTags({ intent }) {
  if (!intent) return null
  return (
    <div className="row" style={{ gap: 6, marginBottom: 10 }}>
      <span className="tag tag-brand">任务: {intent.task}</span>
      {(intent.entities || []).map((e, i) => <span key={i} className="tag">{e}</span>)}
      {intent.time && <span className="tag tag-success">时间: {intent.time}</span>}
      {intent.risk_note && <span className="tag tag-warn">风险: {intent.risk_note}</span>}
    </div>
  )
}

function BotMessage({ msg, onFillQuestion }) {
  const meta = TYPE_META[msg.type] || TYPE_META.answer
  return (
    <div className="chat-row">
      <div className="chat-bubble chat-bot" style={{ width: '100%', maxWidth: '88%' }}>
        <div className="row" style={{ gap: 8, marginBottom: 8 }}>
          <span className={`tag ${meta.cls}`}>{meta.icon} {meta.label}</span>
        </div>
        <IntentTags intent={msg.intent} />
        <div className="answer-text">{msg.answer}</div>
        {msg.type === 'clarification' && (
          <button className="btn btn-sm" style={{ marginTop: 10 }}
            onClick={() => onFillQuestion(msg.clarification_question || msg.answer || '')}>
            补充后重新提问
          </button>
        )}
        {(msg.sources || []).length > 0 && (
          <details style={{ marginTop: 12 }}>
            <summary>参考来源（{msg.sources.length}）</summary>
            <div className="stack" style={{ gap: 8, marginTop: 8 }}>
              {msg.sources.map((s, i) => (
                <div key={i} className="glass-panel" style={{ padding: '10px 12px', borderRadius: 10 }}>
                  <small className="muted">📄 {s.document_name}</small>
                  <p style={{ fontSize: '0.88rem', marginTop: 4 }}>{s.content_snippet}</p>
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

export default function Chat() {
  const [searchParams] = useSearchParams()
  const [question, setQuestion] = useState('')
  const [messages, setMessages] = useState([])  // {role:'user'|'bot', ...}
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const bottomRef = useRef(null)

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, loading])

  const ask = async (text) => {
    if (!text.trim() || loading) return
    setError('')
    setMessages(m => [...m, { role: 'user', text }])
    setQuestion('')
    setLoading(true)
    try {
      const data = await submitQuery(text, getConversationId())
      setMessages(m => [...m, {
        role: 'bot',
        type: data.response_type || 'answer',
        answer: data.answer || data.clarification_question || '',
        clarification_question: data.clarification_question,
        intent: data.intent,
        sources: data.sources,
      }])
    } catch (err) {
      setError(err.message)
      setMessages(m => m.slice(0, -1))  // 回滚失败的用户消息，便于重试
      setQuestion(text)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { const q = searchParams.get('q'); if (q) ask(q) }, [])

  const handleSubmit = (e) => { e.preventDefault(); ask(question) }

  return (
    <div>
      <h1 className="page-title">智能提问</h1>
      <p className="page-subtitle">基于多跳检索与证据校验的企业知识问答，支持多轮上下文理解</p>

      {error && <ErrorMessage message={error} onRetry={() => ask(question)} />}

      <div className="glass-panel" style={{ padding: 18, minHeight: 300, marginBottom: 16 }}>
        {messages.length === 0 && !loading && (
          <div className="empty-state">
            <div className="icon">💬</div>
            <p>开始你的第一次提问吧</p>
            <div className="row" style={{ justifyContent: 'center', marginTop: 16 }}>
              {['胖东来的员工休假制度是怎样的？', '新员工培训计划包含哪些内容？', '温暖基金的申请标准是什么？'].map(s => (
                <span key={s} className="tag" style={{ cursor: 'pointer' }} onClick={() => ask(s)}>{s}</span>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) => m.role === 'user'
          ? <div key={i} className="chat-row"><div className="chat-bubble chat-user">{m.text}</div></div>
          : <BotMessage key={i} msg={m} onFillQuestion={setQuestion} />)}
        {loading && (
          <div className="chat-row">
            <div className="chat-bubble chat-bot typing"><span></span><span></span><span></span></div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <form onSubmit={handleSubmit} className="glass-panel" style={{ padding: 14, display: 'flex', gap: 10, alignItems: 'flex-end' }}>
        <textarea
          rows={2}
          placeholder="输入你的问题，Enter 发送 / Shift+Enter 换行"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); ask(question) } }}
          style={{ flex: 1 }}
        />
        <button className="btn btn-primary" type="submit" disabled={loading || !question.trim()}>
          {loading ? '思考中' : '发送 ↑'}
        </button>
      </form>
    </div>
  )
}
