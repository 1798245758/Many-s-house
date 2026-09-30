import { useState, useEffect } from 'react'
import { getProfile, updateProfile, getSettings, updateSettings } from '../../api/profile'
import { useTheme } from '../../contexts/ThemeContext'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

export default function Profile() {
  const { theme, setTheme } = useTheme()
  const [nickname, setNickname] = useState('')
  const [email, setEmail] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    (async () => {
      try {
        const [profile, settings] = await Promise.all([getProfile(), getSettings()])
        setNickname(profile.nickname || '')
        setEmail(profile.email || '')
        setApiKey(settings.api_key || '')
      } catch (e) { setError(e.message) }
      finally { setLoading(false) }
    })()
  }, [])

  const saveProfile = async () => {
    setSaving(true); setMessage(''); setError('')
    try { await updateProfile({ nickname, email }); setMessage('个人资料已更新') }
    catch (e) { setError(e.message) }
    finally { setSaving(false) }
  }

  const saveSettings = async () => {
    setSaving(true); setMessage(''); setError('')
    try { await updateSettings({ theme, api_key: apiKey || undefined }); setMessage('设置已更新') }
    catch (e) { setError(e.message) }
    finally { setSaving(false) }
  }

  if (loading) return <Loading text="加载资料" />

  return (
    <div style={{ maxWidth: 680, margin: '0 auto' }}>
      <h1 className="page-title">个人信息</h1>
      <p className="page-subtitle">管理你的资料、API Key 与界面主题</p>
      {message && <div className="success-message">✅ {message}</div>}
      {error && <ErrorMessage message={error} />}

      <div className="card">
        <h3 style={{ marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>👤 基本资料</h3>
        <div style={{ marginBottom: 14 }}>
          <label>昵称</label>
          <input value={nickname} onChange={e => setNickname(e.target.value)} placeholder="你的昵称" />
        </div>
        <div style={{ marginBottom: 16 }}>
          <label>邮箱</label>
          <input value={email} onChange={e => setEmail(e.target.value)} placeholder="you@company.com" />
        </div>
        <button className="btn btn-primary" onClick={saveProfile} disabled={saving}>{saving ? '保存中...' : '保存资料'}</button>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>🔑 API Key 配置</h3>
        <div style={{ marginBottom: 16 }}>
          <label>DeepSeek API Key</label>
          <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="sk-..." className="mono" />
        </div>
        <button className="btn btn-primary" onClick={saveSettings} disabled={saving}>{saving ? '保存中...' : '保存密钥'}</button>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 14, display: 'flex', alignItems: 'center', gap: 8 }}>🎨 主题设置</h3>
        <div className="row" style={{ gap: 12 }}>
          {[{ id: 'light', label: '☀️ 亮色' }, { id: 'dark', label: '🌙 暗色' }].map(t => (
            <button key={t.id}
              className={`btn ${theme === t.id ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setTheme(t.id)} style={{ flex: 1 }}>
              {t.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
