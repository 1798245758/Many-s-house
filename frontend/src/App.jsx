import { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import RoleSelect from './components/RoleSelect'
import Home from './pages/Home'
import Chat from './pages/Chat'
import Documents from './pages/Documents'
import History from './pages/History'
import Profile from './pages/Profile'
import AboutData from './pages/AboutData'

export default function App() {
  // 首次进入未选角色时展示角色选择页（localStorage 已有则直接进入）
  const [role, setRoleState] = useState(localStorage.getItem('app_role'))
  if (!role) {
    return <RoleSelect onSelect={setRoleState} />
  }
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/documents" element={<Documents />} />
        <Route path="/history" element={<History />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/about" element={<AboutData />} />
      </Routes>
    </Layout>
  )
}
