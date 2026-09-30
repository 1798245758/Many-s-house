import { useState, useEffect, useCallback, useRef } from 'react'
import { getDocuments, uploadDocument, deleteDocument, replaceDocument, updateVisibility, cancelTask, streamTask } from '../../api/documents'
import { getRole } from '../../api'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

const STATUS_LABELS = { pending: '待处理', processing: '处理中', ready: '就绪', error: '失败', cancelled: '已取消' }
// 异步任务状态徽章：与后端六态状态机对齐（终态卡片在 done 事件后自动移除）
const TASK_STATUS_META = {
  queued: { label: '排队中', cls: '' },
  running: { label: '处理中', cls: 'tag-brand' },
  blocked: { label: '等待重试', cls: 'tag-warn' },
  succeeded: { label: '成功', cls: 'tag-success' },
  failed: { label: '失败', cls: 'tag-danger' },
  cancelled: { label: '已取消', cls: '' },
}
const FILE_ICONS = { pdf: '📕', doc: '📘', docx: '📘', ppt: '📙', pptx: '📙', txt: '📄', md: '📝', xmind: '🧠', png: '🖼️', jpg: '🖼️', jpeg: '🖼️' }
const PAGE_SIZE = 10

export default function Documents() {
  const isManager = getRole() === 'manager'
  const [data, setData] = useState({ items: [], total: 0, page: 1 })
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')
  const [taskCards, setTaskCards] = useState([])
  const streamsRef = useRef({})
  const attachedRef = useRef(new Set())

  const fetchDocs = useCallback(async (page = 1, searchText = '') => {
    setLoading(true); setError('')
    try { setData(await getDocuments(page, PAGE_SIZE, searchText)) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetchDocs(1, search) }, [fetchDocs, search])

  const startStreaming = useCallback((taskId) => {
    if (streamsRef.current[taskId]) return
    streamsRef.current[taskId] = streamTask(taskId, {
      onProgress: (p) => setTaskCards(cs => cs.map(c => c.task_id === taskId ? { ...c, ...p } : c)),
      onDone: () => {
        delete streamsRef.current[taskId]
        setTaskCards(cs => cs.filter(c => c.task_id !== taskId))
        fetchDocs(1, '')
      },
      onError: () => { delete streamsRef.current[taskId] },
    })
  }, [fetchDocs])

  useEffect(() => () => {
    Object.values(streamsRef.current).forEach(es => es.close())
    streamsRef.current = {}
  }, [])

  useEffect(() => {
    data.items.forEach(doc => {
      if (doc.status === 'processing' && doc.task_id && !attachedRef.current.has(doc.task_id)) {
        attachedRef.current.add(doc.task_id)
        setTaskCards(cs => cs.some(c => c.task_id === doc.task_id) ? cs
          : [{ task_id: doc.task_id, filename: doc.filename, status: 'queued', progress: 0, message: '恢复进度连接…' }, ...cs])
        startStreaming(doc.task_id)
      }
    })
  }, [data, startStreaming])

  const handleSearch = () => setSearch(searchInput)

  const handleUpload = async (e) => {
    const files = e.target.files
    if (!files.length) return
    setUploading(true); setError('')
    try {
      const result = await uploadDocument(files)
      const cards = (result.tasks || []).map(t => ({ task_id: t.task_id, filename: t.filename, status: 'queued', progress: 0, message: '任务已创建，等待执行' }))
      if (cards.length) {
        setTaskCards(cs => [...cards, ...cs])
        cards.forEach(c => { attachedRef.current.add(c.task_id); startStreaming(c.task_id) })
      }
      if ((result.errors || []).length) setError(result.errors.map(x => `${x.filename}: ${x.message}`).join('；'))
      fetchDocs(1, search)
    } catch (err) { setError(err.message) }
    finally { setUploading(false); e.target.value = '' }
  }

  const handleCancelTask = async (taskId) => {
    try { await cancelTask(taskId) } catch (err) { setError(err.message) }
  }
  const handleDelete = async (id) => {
    try { await deleteDocument(id); fetchDocs(data.page, search) } catch (err) { setError(err.message) }
  }
  const handleReplace = async (id, file) => {
    try {
      const result = await replaceDocument(id, file)
      const card = { task_id: result.task_id, filename: file.name, status: 'queued', progress: 0, message: '替换任务已创建，等待执行' }
      attachedRef.current.add(card.task_id)
      setTaskCards(cs => [card, ...cs])
      startStreaming(card.task_id)
    } catch (err) { setError(err.message) }
  }
  const handleToggleVisibility = async (doc) => {
    const next = doc.visibility === 'manager_only' ? 'all' : 'manager_only'
    try { await updateVisibility(doc.id, next); fetchDocs(data.page, search) } catch (err) { setError(err.message) }
  }

  const totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE))

  return (
    <div>
      <h1 className="page-title">文档管理</h1>
      <p className="page-subtitle">上传企业文档构建专属知识库，实时查看异步入库进度</p>

      <div className="row" style={{ marginBottom: 16 }}>
        <input value={searchInput} onChange={e => setSearchInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSearch()} placeholder="🔍 搜索文件名..." style={{ maxWidth: 280 }} />
        <button className="btn btn-primary" onClick={handleSearch}>搜索</button>
        {isManager ? (
          <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
            {uploading ? '上传中...' : '⬆ 上传文档'}
            <input type="file" multiple accept=".pdf,.txt,.md,.xmind,.png,.jpg,.jpeg,.gif,.webp,.bmp" onChange={handleUpload} hidden disabled={uploading} />
          </label>
        ) : <span className="muted" style={{ fontSize: '0.85rem' }}>员工角色仅可查看文档，文档维护由经理操作</span>}
      </div>

      {error && <ErrorMessage message={error} />}

      {taskCards.length > 0 && (
        <div className="stack" style={{ marginBottom: 16 }}>
          {taskCards.map(card => {
            const meta = TASK_STATUS_META[card.status] || TASK_STATUS_META.running
            return (
              <div key={card.task_id} className="card fade-in" style={{ padding: '14px 18px', marginBottom: 0 }}>
                <div className="row-between">
                  <strong style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>⚙️ {card.filename}</strong>
                  <div className="row" style={{ gap: 8, flexShrink: 0 }}>
                    <span className={`tag ${meta.cls}`}>{meta.label}</span>
                    <span className="muted" style={{ fontSize: '0.8rem', minWidth: 36, textAlign: 'right' }}>{card.progress || 0}%</span>
                    <button className="btn btn-sm" onClick={() => handleCancelTask(card.task_id)}>取消</button>
                  </div>
                </div>
                <div className="progress-track" style={{ marginTop: 10 }}>
                  <div className="progress-bar" style={{ width: `${card.progress || 0}%` }} />
                </div>
                <div className="muted" style={{ fontSize: '0.8rem', marginTop: 8 }}>{card.error || card.message}</div>
              </div>
            )
          })}
        </div>
      )}

      {loading ? <Loading text="加载文档" /> : data.items.length === 0 ? (
        <div className="empty-state">
          <div className="icon">📭</div>
          <p>{search ? '没有匹配的文档' : '暂无文档，请上传'}</p>
        </div>
      ) : (
        <>
          <div className="stack stagger">
            {data.items.map(doc => (
              <div key={doc.id} className="card card-hover row-between" style={{ marginBottom: 0 }}>
                <div className="row" style={{ overflow: 'hidden', marginRight: 12, flexWrap: 'nowrap' }}>
                  <div className="card-icon" style={{ width: 42, height: 42, fontSize: '1.3rem', margin: 0, flexShrink: 0 }}>
                    {FILE_ICONS[doc.file_type] || '📄'}
                  </div>
                  <div style={{ overflow: 'hidden' }}>
                    <strong style={{ wordBreak: 'break-all' }}>{doc.filename}</strong>
                    {isManager && doc.visibility === 'manager_only' && <span className="tag tag-warn" style={{ marginLeft: 8 }}>经理专属</span>}
                    <div className="muted" style={{ fontSize: '0.82rem', marginTop: 4 }}>
                      {doc.file_type.toUpperCase()} · {doc.file_size > 1024 ? `${(doc.file_size / 1024).toFixed(1)}KB` : `${doc.file_size}B`} · {doc.chunk_count} 块 · {STATUS_LABELS[doc.status] || doc.status}
                    </div>
                  </div>
                </div>
                {isManager && (
                  <div className="row" style={{ gap: 8, flexShrink: 0 }}>
                    {(doc.status === 'error' || doc.status === 'cancelled') && doc.task_id && (
                      <button className="btn btn-sm" onClick={() => window.open(`/diagnosis?task_id=${encodeURIComponent(doc.task_id)}`, '_blank')}>🔍 诊断</button>
                    )}
                    <button className="btn btn-sm" onClick={() => handleToggleVisibility(doc)}>
                      {doc.visibility === 'manager_only' ? '设为全员可见' : '设为经理专属'}
                    </button>
                    <label className="btn btn-sm btn-primary" style={{ cursor: 'pointer' }}>
                      替换
                      <input type="file" accept=".pdf,.txt,.md,.xmind,.png,.jpg,.jpeg,.gif,.webp,.bmp" onChange={e => { const f = e.target.files[0]; if (f) handleReplace(doc.id, f) }} hidden />
                    </label>
                    <button className="btn btn-sm btn-danger" onClick={() => handleDelete(doc.id)}>删除</button>
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="row" style={{ justifyContent: 'center', marginTop: 20 }}>
            <button className="btn" disabled={data.page <= 1} onClick={() => fetchDocs(data.page - 1, search)}>← 上一页</button>
            <span className="muted" style={{ fontSize: '0.9rem' }}>第 {data.page} / {totalPages} 页（共 {data.total} 条）</span>
            <button className="btn" disabled={data.page >= totalPages} onClick={() => fetchDocs(data.page + 1, search)}>下一页 →</button>
          </div>
        </>
      )}
    </div>
  )
}
