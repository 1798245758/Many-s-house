import { get, post, del } from './index'

// 长期记忆管理：列表 / 手动新增 / 删除
export function getMemories() { return get('/api/memories') }
export function createMemory(content, category = 'preference') {
  return post('/api/memories', { content, category })
}
export function deleteMemory(key) { return del(`/api/memories/${key}`) }
