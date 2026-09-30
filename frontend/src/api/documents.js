import { get, del, patch, post, uploadFile, getRole } from './index'
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
export function updateVisibility(id, visibility) { return patch(`/api/documents/${id}/visibility`, { visibility }) }

// 异步任务：快照查询 / 取消 / SSE 进度推流（EventSource 无法设请求头，角色走查询参数）
export function getTask(taskId) { return get(`/api/tasks/${taskId}`) }
export function cancelTask(taskId) { return post(`/api/tasks/${taskId}/cancel`) }
export function streamTask(taskId, { onProgress, onDone, onError } = {}) {
  const es = new EventSource(`/api/tasks/${taskId}/stream?role=${getRole()}`)
  es.addEventListener('progress', (e) => onProgress && onProgress(JSON.parse(e.data)))
  es.addEventListener('done', (e) => { es.close(); onDone && onDone(JSON.parse(e.data)) })
  es.onerror = () => { es.close(); onError && onError() }
  return es
}
