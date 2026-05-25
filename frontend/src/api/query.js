import { post } from './index'
export function submitQuery(question) { return post('/api/query', { question }) }
