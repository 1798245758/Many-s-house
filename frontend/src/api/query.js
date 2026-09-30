import { post } from './index'
export function submitQuery(question, conversationId) {
  return post('/api/query', { question, conversation_id: conversationId })
}
