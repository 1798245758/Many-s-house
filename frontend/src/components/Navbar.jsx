import { NavLink, Link } from 'react-router-dom'
import { getRole, clearRole } from '../api'
import { useTheme } from '../contexts/ThemeContext'

const LINKS = [
  { to: '/', label: '首页', end: true },
  { to: '/chat', label: '提问' },
  { to: '/documents', label: '文档' },
  { to: '/history', label: '历史' },
  { to: '/memory', label: '记忆' },
  { to: '/profile', label: '个人' },
  { to: '/about', label: '关于' },
]

export default function Navbar() {
  const role = getRole()
  const { theme, toggleTheme } = useTheme()
  const switchRole = () => { clearRole(); window.location.reload() }

  return (
    <nav className="navbar">
      <Link to="/" className="brand">
        <span className="logo"><span>◆</span></span>
        <span className="brand-text">企业问答助手</span>
      </Link>

      {LINKS.map(l => (
        <NavLink key={l.to} to={l.to} end={l.end}
          className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          {l.label}
        </NavLink>
      ))}
      {role === 'manager' && (
        <NavLink to="/diagnosis" className={({ isActive }) => `nav-link${isActive ? ' active' : ''}`}>
          诊断
        </NavLink>
      )}

      <span className="nav-spacer" />

      <button className="theme-toggle" onClick={toggleTheme} title="切换主题"
        aria-label="切换主题">
        {theme === 'dark' ? '☀️' : '🌙'}
      </button>
      <span className={`role-chip${role === 'manager' ? ' manager' : ''}`}
        onClick={switchRole} title="点击切换角色">
        {role === 'manager' ? '💼 经理' : '👤 员工'}
      </span>
    </nav>
  )
}
