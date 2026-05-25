import { get, del, uploadFile } from './index'
export function getDocuments() { return get('/api/documents') }
export function uploadDocument(fileList) {
  const fd = new FormData()
  const files = Array.isArray(fileList) ? fileList : [fileList]
  files.forEach(f => fd.append('files', f))
  return uploadFile('/api/documents/upload', fd)
}
export function deleteDocument(id) { return del(`/api/documents/${id}`) }
