import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { submitQuery } from '../../api/query'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

const tagBase = {display:'inline-block',padding:'2px 8px',borderRadius:12,fontSize:'0.8rem',marginRight:6,marginBottom:4}

function IntentTags({ intent }) {
  if (!intent) return null
  return (
    <div style={{marginBottom:10}}>
      <small style={{color:'#666',marginRight:8}}>意图分析:</small>
      <span style={{...tagBase,background:'#e6f0ff',color:'#1a56db'}}>任务: {intent.task}</span>
      {(intent.entities || []).map((e, i) => (
        <span key={i} style={{...tagBase,background:'#f0f0f0',color:'#333'}}>{e}</span>
      ))}
      {intent.time && <span style={{...tagBase,background:'#e6fff0',color:'#0a7a4b'}}>时间: {intent.time}</span>}
      {intent.risk_note && <span style={{...tagBase,background:'#fff3e0',color:'#b45309'}}>风险: {intent.risk_note}</span>}
    </div>
  )
}

export default function Chat() {
  const [searchParams] = useSearchParams()
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const q = searchParams.get('q')
    if (q) {
      setQuestion(q)
      handleAsk(q)
    }
  }, [])

  const handleAsk = async (text) => {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await submitQuery(text)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    handleAsk(question)
  }

  const type = result?.response_type || 'answer'

  return (
    <div>
      <h1 className="page-title">提问</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={3}
          placeholder="输入你的问题，例如：胖东来的员工休假制度是怎样的？"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          style={{marginBottom:12}}
        />
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? '查询中...' : '发送'}
        </button>
      </form>
      {error && <ErrorMessage message={error} onRetry={() => handleSubmit({ preventDefault: () => {} })} />}
      {loading && <Loading />}
      {result && type === 'clarification' && (
        <div className="card" style={{marginTop:20,border:'1px solid #f59e0b',background:'#fffbeb'}}>
          <IntentTags intent={result.intent} />
          <h3>需要补充信息</h3>
          <div className="answer-text">{result.clarification_question || result.answer}</div>
          <button
            className="btn"
            style={{marginTop:10}}
            onClick={() => setQuestion(result.clarification_question || result.answer || '')}
          >
            补充后重新提问
          </button>
        </div>
      )}
      {result && type === 'refusal' && (
        <div className="card" style={{marginTop:20,border:'1px solid #ef4444',background:'#fef2f2'}}>
          <h3>无法回答</h3>
          <div className="answer-text" style={{color:'#b91c1c'}}>{result.answer}</div>
        </div>
      )}
      {result && type === 'answer' && (
        <div className="card" style={{marginTop:20}}>
          <IntentTags intent={result.intent} />
          <h3>回答</h3>
          <div className="answer-text">{result.answer}</div>
          {(result.sources || []).length > 0 && (
            <details>
              <summary>参考来源 ({result.sources.length})</summary>
              {result.sources.map((s, i) => (
                <div key={i} style={{marginTop:8,padding:8,background:'#f5f5f5',borderRadius:4}}>
                  <small style={{color:'#666'}}>{s.document_name}</small>
                  <p style={{fontSize:'0.9rem'}}>{s.content_snippet}</p>
                </div>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  )
}
