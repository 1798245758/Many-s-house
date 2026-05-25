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
    setSaving(true)
    setMessage('')
    setError('')
    try {
      await updateProfile({ nickname, email })
      setMessage('个人资料已更新')
    } catch (e) { setError(e.message) }
    finally { setSaving(false) }
  }

  const saveSettings = async () => {
    setSaving(true)
    setMessage('')
    setError('')
    try {
      await updateSettings({ theme, api_key: apiKey || undefined })
      setMessage('设置已更新')
    } catch (e) { setError(e.message) }
    finally { setSaving(false) }
  }

  if (loading) return <Loading />

  return (
    <div>
      <h1 className="page-title">个人信息</h1>
      {message && <div className="card" style={{borderColor:'#38a169',color:'#38a169'}}>{message}</div>}
      {error && <ErrorMessage message={error} />}

      <div className="card">
        <h3 style={{marginBottom:12}}>基本资料</h3>
        <div style={{marginBottom:12}}>
          <label>昵称</label>
          <input value={nickname} onChange={e => setNickname(e.target.value)} />
        </div>
        <div style={{marginBottom:12}}>
          <label>邮箱</label>
          <input value={email} onChange={e => setEmail(e.target.value)} />
        </div>
        <button className="btn btn-primary" onClick={saveProfile} disabled={saving}>保存</button>
      </div>

      <div className="card">
        <h3 style={{marginBottom:12}}>API Key 配置</h3>
        <div style={{marginBottom:12}}>
          <label>DeepSeek API Key</label>
          <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="sk-..." />
        </div>
        <button className="btn btn-primary" onClick={saveSettings} disabled={saving}>保存</button>
      </div>

      <div className="card">
        <h3 style={{marginBottom:12}}>主题设置</h3>
        <div>
          <label><input type="radio" checked={theme === 'light'} onChange={() => setTheme('light')} /> 亮色</label>
          <label style={{marginLeft:16}}><input type="radio" checked={theme === 'dark'} onChange={() => setTheme('dark')} /> 暗色</label>
        </div>
      </div>
    </div>
  )
}
