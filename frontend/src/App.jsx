import { Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Home from './pages/Home'
import Chat from './pages/Chat'
import Documents from './pages/Documents'
import History from './pages/History'
import Profile from './pages/Profile'
import AboutData from './pages/AboutData'

export default function App() {
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
