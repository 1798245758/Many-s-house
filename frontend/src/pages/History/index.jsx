import { useNavigate } from 'react-router-dom'
import { useState, useEffect, useCallback } from 'react'
import { getHistories, deleteHistory } from '../../api/history'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

export default function History() {
  const navigate = useNavigate()
  const [histories, setHistories] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetch = useCallback(async () => {
    setLoading(true); setError('')
    try { setHistories(await getHistories()) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetch() }, [fetch])

  const handleDelete = async (id) => {
    try { await deleteHistory(id); fetch() }
    catch (err) { setError(err.message) }
  }

  return (
    <div>
      <h1 className="page-title">历史记录</h1>
      <p className="page-subtitle">你的查询足迹，可随时回溯或重新提问</p>
      {error && <ErrorMessage message={error} />}
      {loading ? <Loading text="加载历史" /> : histories.length === 0 ? (
        <div className="empty-state">
          <div className="icon">🕘</div>
          <p>暂无历史记录</p>
        </div>
      ) : (
        <div className="stack stagger">
          {histories.map(h => (
            <div key={h.id} className="card card-hover" style={{ marginBottom: 0 }}>
              <div className="row-between" style={{ alignItems: 'flex-start' }}>
                <div style={{ flex: 1, overflow: 'hidden' }}>
                  <strong style={{ color: 'var(--primary)' }}>Q: {h.query_text}</strong>
                  <p className="muted" style={{ marginTop: 8, whiteSpace: 'pre-wrap', fontSize: '0.92rem' }}>
                    {h.answer_text.slice(0, 300)}{h.answer_text.length > 300 ? '...' : ''}
                  </p>
                  <small className="muted">🕒 {h.created_at}</small>
                </div>
                <div className="row" style={{ gap: 8, flexShrink: 0 }}>
                  <button className="btn btn-sm btn-primary" onClick={() => navigate(`/chat?q=${encodeURIComponent(h.query_text)}`)}>重新提问</button>
                  <button className="btn btn-sm btn-danger" onClick={() => handleDelete(h.id)}>删除</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
