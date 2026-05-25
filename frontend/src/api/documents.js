import { get, del, uploadFile } from './index'
export function getDocuments() { return get('/api/documents') }
export function uploadDocument(file) {
  const fd = new FormData()
  fd.append('file', file)
  return uploadFile('/api/documents/upload', fd)
}
export function deleteDocument(id) { return del(`/api/documents/${id}`) }
