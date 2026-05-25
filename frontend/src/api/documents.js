import { get, del, uploadFile } from './index'
export function getDocuments() { return get('/api/documents') }
export function uploadDocument(fileList) {
  const fd = new FormData()
  for (const file of fileList) {
    fd.append('files', file)
  }
  return uploadFile('/api/documents/upload', fd)
}
export function deleteDocument(id) { return del(`/api/documents/${id}`) }
