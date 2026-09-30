import { Link } from 'react-router-dom'
import { getRole } from '../../api'

const FEATURES = [
  { to: '/chat', icon: '💬', title: '智能问答', desc: '多跳检索 + 证据校验，制度、培训、管理问题即问即答' },
  { to: '/documents', icon: '📚', title: '文档管理', desc: '支持 PDF / Word / PPT / TXT 等格式，一键构建专属知识库' },
  { to: '/history', icon: '🕘', title: '历史记录', desc: '查询足迹随时回溯，一键复现历史提问' },
]

export default function Home() {
  const isManager = getRole() === 'manager'
  const features = isManager
    ? [...FEATURES, { to: '/diagnosis', icon: '🩺', title: '错误诊断', desc: '基于 CheckPoint 现场证据，大模型多轮追问定位失败根因' }]
    : FEATURES

  return (
    <div>
      <div className="hero">
        <div className="hero-badge"><span className="dot" /> DeepSeek 驱动 · 本地知识库</div>
        <h1>企业问答助手</h1>
        <p>内置胖东来企业文化、员工手册、管理制度等知识，也支持上传企业文档构建专属知识库，随时咨询制度、培训、管理等问题。</p>
        <div className="row" style={{ justifyContent: 'center' }}>
          <Link to="/chat" className="btn btn-primary">开始提问 →</Link>
          <Link to="/documents" className="btn btn-ghost">上传文档</Link>
        </div>
      </div>

      <div className="feature-grid stagger">
        {features.map(f => (
          <Link key={f.to} to={f.to} className="card card-interactive card-hover"
            style={{ textDecoration: 'none', color: 'inherit' }}>
            <div className="card-icon">{f.icon}</div>
            <h3>{f.title}</h3>
            <p className="muted" style={{ marginTop: 8, fontSize: '0.9rem' }}>{f.desc}</p>
          </Link>
        ))}
      </div>
    </div>
  )
}
