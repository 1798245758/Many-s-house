import { useState } from 'react'
import { submitQuery } from '../../api/query'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

export default function Chat() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError('')
    setAnswer('')
    try {
      const data = await submitQuery(question)
      setAnswer(data.answer)
      setSources(data.sources || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1 className="page-title">提问</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={3}
          placeholder="输入你的问题..."
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
      {answer && (
        <div className="card" style={{marginTop:20}}>
          <h3>回答</h3>
          <p style={{whiteSpace:'pre-wrap',margin:'12px 0'}}>{answer}</p>
          {sources.length > 0 && (
            <details>
              <summary>参考来源 ({sources.length})</summary>
              {sources.map((s, i) => (
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
