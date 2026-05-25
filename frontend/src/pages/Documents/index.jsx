import { useState, useEffect, useCallback } from 'react'
import { getDocuments, uploadDocument, deleteDocument } from '../../api/documents'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

const STATUS_LABELS = { pending: '待处理', processing: '处理中', ready: '就绪', error: '失败' }

export default function Documents() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  const fetchDocs = useCallback(async () => {
    setLoading(true)
    setError('')
    try { setDocs(await getDocuments()) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetchDocs() }, [fetchDocs])

  const handleUpload = async (e) => {
    const files = e.target.files
    if (!files.length) return
    setUploading(true)
    setError('')
    try { await uploadDocument(files); fetchDocs() }
    catch (err) { setError(err.message) }
    finally { setUploading(false) }
  }

  const handleDelete = async (id) => {
    try { await deleteDocument(id); fetchDocs() }
    catch (err) { setError(err.message) }
  }

  return (
    <div>
      <h1 className="page-title">文档管理</h1>
      <div style={{marginBottom:20}}>
        <label className="btn btn-primary" style={{cursor:'pointer'}}>
          {uploading ? '上传中...' : '上传文档'}
          <input type="file" multiple accept=".pdf,.txt,.md,.png,.jpg,.jpeg,.gif,.webp,.bmp" onChange={handleUpload} hidden disabled={uploading} />
        </label>
      </div>
      {error && <ErrorMessage message={error} />}
      {loading ? <Loading /> : docs.length === 0 ? (
        <p style={{color:'#999',padding:'40px 0',textAlign:'center'}}>暂无文档，请上传</p>
      ) : docs.map(doc => (
        <div key={doc.id} className="card" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <div>
            <strong>{doc.filename}</strong>
            <div style={{color:'#999',fontSize:'0.85rem',marginTop:4}}>
              {doc.file_type.toUpperCase()} · {doc.file_size > 1024 ? `${(doc.file_size/1024).toFixed(1)}KB` : `${doc.file_size}B`} · {doc.chunk_count} 块 · {STATUS_LABELS[doc.status] || doc.status}
            </div>
          </div>
          <button className="btn btn-danger" onClick={() => handleDelete(doc.id)}>删除</button>
        </div>
      ))}
    </div>
  )
}
