import { Link } from 'react-router-dom'
import { getRole, clearRole } from '../api'

export default function Navbar() {
  const role = getRole()
  const switchRole = () => {
    clearRole()
    window.location.reload()
  }
  return (
    <nav className="navbar">
      <Link to="/">首页</Link>
      <Link to="/chat">提问</Link>
      <Link to="/documents">文档</Link>
      <Link to="/history">历史</Link>
      <Link to="/profile">个人</Link>
      <Link to="/about">关于</Link>
      <span
        onClick={switchRole}
        title="点击切换角色"
        style={{
          marginLeft: 'auto', cursor: 'pointer', fontSize: '0.85rem',
          padding: '2px 10px', borderRadius: 12,
          background: role === 'manager' ? '#e6f0ff' : '#f0f0f0',
          color: role === 'manager' ? '#1a56db' : '#333',
        }}
      >
        {role === 'manager' ? '💼 经理' : '👤 员工'}（点击切换）
      </span>
    </nav>
  )
}
