import { Link } from 'react-router-dom'
export default function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/">首页</Link>
      <Link to="/chat">提问</Link>
      <Link to="/documents">文档</Link>
      <Link to="/history">历史</Link>
      <Link to="/profile">个人</Link>
      <Link to="/about">关于</Link>
    </nav>
  )
}
