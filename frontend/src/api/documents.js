import { get, del, uploadFile } from './index'
export function getDocuments(page = 1, pageSize = 10, search = '') {
  const params = new URLSearchParams({ page, page_size: pageSize, search })
  return get(`/api/documents?${params}`)
}
export function uploadDocument(fileList) {
  const fd = new FormData()
  for (const file of fileList) {
    fd.append('files', file)
  }
  return uploadFile('/api/documents/upload', fd)
}
export function replaceDocument(id, file) {
  const fd = new FormData()
  fd.append('file', file)
  return uploadFile(`/api/documents/${id}`, fd, 'PUT')
}
export function deleteDocument(id) { return del(`/api/documents/${id}`) }
