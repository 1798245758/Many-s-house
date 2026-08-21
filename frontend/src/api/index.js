const BASE_URL = ''

// 角色管理：前端自选、无账户验证；缺省/非法一律按员工（最小权限）
const ROLE_KEY = 'app_role'
export function getRole() { return localStorage.getItem(ROLE_KEY) === 'manager' ? 'manager' : 'employee' }
export function setRole(role) { localStorage.setItem(ROLE_KEY, role) }
export function clearRole() { localStorage.removeItem(ROLE_KEY) }

async function request(url, options = {}) {
  let res
  try {
    res = await fetch(`${BASE_URL}${url}`, {
      headers: { 'Content-Type': 'application/json', 'X-Role': getRole(), ...options.headers },
      ...options,
    })
  } catch {
    throw new Error('后端服务未启动，请在 frontend/ 目录执行 npm run dev 自动启动')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.message || '服务器错误，请稍后重试')
  }
  const data = await res.json()
  if (data.code !== 'SUCCESS') {
    throw new Error(data.message || '请求失败')
  }
  return data.data
}

export async function get(url) { return request(url) }
export async function post(url, body) { return request(url, { method: 'POST', body: JSON.stringify(body) }) }
export async function put(url, body) { return request(url, { method: 'PUT', body: JSON.stringify(body) }) }
export async function patch(url, body) { return request(url, { method: 'PATCH', body: JSON.stringify(body) }) }
export async function del(url) { return request(url, { method: 'DELETE' }) }
export async function uploadFile(url, formData, method = 'POST') {
  let res
  try {
    res = await fetch(`${BASE_URL}${url}`, { method, body: formData, headers: { 'X-Role': getRole() } })
  } catch {
    throw new Error('后端服务未启动，请在 frontend/ 目录执行 npm run dev 自动启动')
  }
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.message || '上传失败')
  }
  const data = await res.json()
  if (data.code !== 'SUCCESS') throw new Error(data.message || '上传失败')
  return data.data
}
