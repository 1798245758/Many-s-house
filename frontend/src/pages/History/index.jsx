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
    setLoading(true)
    setError('')
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
      {error && <ErrorMessage message={error} />}
      {loading ? <Loading /> : histories.length === 0 ? (
        <p style={{color:'#999',textAlign:'center',padding:'40px 0'}}>暂无历史记录</p>
      ) : histories.map(h => (
        <div key={h.id} className="card">
          <div style={{display:'flex',justifyContent:'space-between'}}>
            <div style={{flex:1}}>
              <strong>Q: {h.query_text}</strong>
              <p style={{marginTop:8,whiteSpace:'pre-wrap',color:'#555'}}>{h.answer_text.slice(0, 300)}{h.answer_text.length > 300 ? '...' : ''}</p>
              <small style={{color:'#999'}}>{h.created_at}</small>
            </div>
              <div style={{display:'flex',gap:8,flexShrink:0}}>
                <button className="btn btn-primary" style={{height:'fit-content'}} onClick={() => navigate(`/chat?q=${encodeURIComponent(h.query_text)}`)}>重新提问</button>
                <button className="btn btn-danger" style={{height:'fit-content'}} onClick={() => handleDelete(h.id)}>删除</button>
              </div>
          </div>
        </div>
      ))}
    </div>
  )
}
