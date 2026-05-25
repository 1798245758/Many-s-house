import { get, del } from './index'
export function getHistories() { return get('/api/history') }
export function deleteHistory(id) { return del(`/api/history/${id}`) }
