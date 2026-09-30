import { useState, useEffect, useCallback } from 'react'
import { getMemories, createMemory, deleteMemory } from '../../api/memory'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

const CATEGORIES = [
  { value: 'role', label: '身份' },
  { value: 'instruction', label: '指令' },
  { value: 'format', label: '格式' },
  { value: 'preference', label: '偏好' },
]
const CAT_LABEL = Object.fromEntries(CATEGORIES.map(c => [c.value, c.label]))

export default function Memory() {
  const [memories, setMemories] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [content, setContent] = useState('')
  const [category, setCategory] = useState('preference')

  const fetch = useCallback(async () => {
    setLoading(true); setError('')
    try { setMemories(await getMemories()) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetch() }, [fetch])

  const handleAdd = async () => {
    if (!content.trim()) return
    try { await createMemory(content.trim(), category); setContent(''); fetch() }
    catch (e) { setError(e.message) }
  }

  const handleDelete = async (key) => {
    try { await deleteMemory(key); fetch() }
    catch (e) { setError(e.message) }
  }

  return (
    <div>
      <h1 className="page-title">长期记忆</h1>
      <p className="page-subtitle">跨会话恒定生效的用户事实（身份 / 指令 / 格式 / 偏好），问答时自动注入</p>
      {error && <ErrorMessage message={error} />}

      <div className="card" style={{ marginBottom: 16 }}>
        <div className="row" style={{ gap: 8, flexWrap: 'wrap' }}>
          <select className="input" value={category} onChange={e => setCategory(e.target.value)}
            style={{ width: 120 }}>
            {CATEGORIES.map(c => <option key={c.value} value={c.value}>{c.label}</option>)}
          </select>
          <input className="input" style={{ flex: 1, minWidth: 200 }} placeholder="例如：回答要简洁 / 用表格回答 / 我是餐饮部经理"
            value={content} onChange={e => setContent(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleAdd()} />
          <button className="btn btn-primary" onClick={handleAdd}>新增</button>
        </div>
      </div>

      {loading ? <Loading text="加载记忆" /> : memories.length === 0 ? (
        <div className="empty-state">
          <div className="icon">🧠</div>
          <p>暂无长期记忆</p>
        </div>
      ) : (
        <div className="stack stagger">
          {memories.map(m => (
            <div key={m.key} className="card card-hover" style={{ marginBottom: 0 }}>
              <div className="row-between" style={{ alignItems: 'flex-start' }}>
                <div style={{ flex: 1, overflow: 'hidden' }}>
                  <span className="role-chip" style={{ marginRight: 8 }}>{CAT_LABEL[m.category] || m.category}</span>
                  <strong>{m.content}</strong>
                  <div><small className="muted">🕒 {m.updated_at}</small></div>
                </div>
                <button className="btn btn-sm btn-danger" style={{ flexShrink: 0 }}
                  onClick={() => handleDelete(m.key)}>删除</button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
