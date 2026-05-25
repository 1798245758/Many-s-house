const BASE_URL = ''

async function request(url, options = {}) {
  const res = await fetch(`${BASE_URL}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.message || '网络连接失败，请检查网络')
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
export async function del(url) { return request(url, { method: 'DELETE' }) }
export async function uploadFile(url, formData) {
  const res = await fetch(`${BASE_URL}${url}`, { method: 'POST', body: formData })
  if (!res.ok) throw new Error('上传失败')
  const data = await res.json()
  if (data.code !== 'SUCCESS') throw new Error(data.message)
  return data.data
}
