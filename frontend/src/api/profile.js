import { get, put } from './index'
export function getProfile() { return get('/api/profile') }
export function updateProfile(data) { return put('/api/profile', data) }
export function getSettings() { return get('/api/settings') }
export function updateSettings(data) { return put('/api/settings', data) }
