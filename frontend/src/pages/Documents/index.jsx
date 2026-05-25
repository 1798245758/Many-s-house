import { useState, useEffect, useCallback } from 'react'
import { getDocuments, uploadDocument, deleteDocument, replaceDocument } from '../../api/documents'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

const STATUS_LABELS = { pending: '待处理', processing: '处理中', ready: '就绪', error: '失败' }
const PAGE_SIZE = 10

export default function Documents() {
  const [data, setData] = useState({ items: [], total: 0, page: 1 })
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')
  const [search, setSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')

  const fetchDocs = useCallback(async (page = 1, searchText = '') => {
    setLoading(true)
    setError('')
    try {
      const result = await getDocuments(page, PAGE_SIZE, searchText)
      setData(result)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetchDocs(1, search) }, [fetchDocs, search])

  const handleSearch = () => {
    setSearch(searchInput)
  }

  const handleUpload = async (e) => {
    const files = e.target.files
    if (!files.length) return
    setUploading(true)
    setError('')
    try { await uploadDocument(files); fetchDocs(1, search) }
    catch (err) { setError(err.message) }
    finally { setUploading(false); e.target.value = '' }
  }

  const handleDelete = async (id) => {
    try { await deleteDocument(id); fetchDocs(data.page, search) }
    catch (err) { setError(err.message) }
  }

  const handleReplace = async (id, file) => {
    try {
      await replaceDocument(id, file)
      fetchDocs(data.page, search)
    } catch (err) { setError(err.message) }
  }

  const totalPages = Math.max(1, Math.ceil(data.total / PAGE_SIZE))

  return (
    <div>
      <h1 className="page-title">文档管理</h1>

      <div style={{display:'flex',gap:12,alignItems:'center',marginBottom:16,flexWrap:'wrap'}}>
        <input
          value={searchInput}
          onChange={e => setSearchInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSearch()}
          placeholder="搜索文件名..."
          style={{maxWidth:260}}
        />
        <button className="btn btn-primary" onClick={handleSearch}>搜索</button>
        <label className="btn btn-primary" style={{cursor:'pointer'}}>
          {uploading ? '上传中...' : '上传文档'}
          <input type="file" multiple accept=".pdf,.txt,.md,.xmind,.png,.jpg,.jpeg,.gif,.webp,.bmp" onChange={handleUpload} hidden disabled={uploading} />
        </label>
      </div>

      {error && <ErrorMessage message={error} />}

      {loading ? <Loading /> : data.items.length === 0 ? (
        <p style={{color:'#999',padding:'40px 0',textAlign:'center'}}>{search ? '没有匹配的文档' : '暂无文档，请上传'}</p>
      ) : (
        <>
          {data.items.map(doc => (
            <div key={doc.id} className="card" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
              <div style={{overflow:'hidden',marginRight:12}}>
                <strong style={{wordBreak:'break-all'}}>{doc.filename}</strong>
                <div style={{color:'#999',fontSize:'0.85rem',marginTop:4}}>
                  {doc.file_type.toUpperCase()} · {doc.file_size > 1024 ? `${(doc.file_size/1024).toFixed(1)}KB` : `${doc.file_size}B`} · {doc.chunk_count} 块 · {STATUS_LABELS[doc.status] || doc.status}
                </div>
              </div>
              <div style={{display:'flex',gap:8,flexShrink:0}}>
                <label className="btn btn-primary" style={{cursor:'pointer',fontSize:'0.85rem'}}>
                  替换
                  <input type="file" accept=".pdf,.txt,.md,.xmind,.png,.jpg,.jpeg,.gif,.webp,.bmp" onChange={e => { const f = e.target.files[0]; if (f) handleReplace(doc.id, f) }} hidden />
                </label>
                <button className="btn btn-danger" onClick={() => handleDelete(doc.id)}>删除</button>
              </div>
            </div>
          ))}

          <div style={{display:'flex',justifyContent:'center',alignItems:'center',gap:8,marginTop:16}}>
            <button className="btn" disabled={data.page <= 1} onClick={() => fetchDocs(data.page - 1, search)}>上一页</button>
            <span style={{fontSize:'0.9rem',color:'#666'}}>第 {data.page} / {totalPages} 页（共 {data.total} 条）</span>
            <button className="btn" disabled={data.page >= totalPages} onClick={() => fetchDocs(data.page + 1, search)}>下一页</button>
          </div>
        </>
      )}
    </div>
  )
}
