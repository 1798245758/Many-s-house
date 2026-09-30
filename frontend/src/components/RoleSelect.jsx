import { setRole } from '../api'

export default function RoleSelect({ onSelect }) {
  const choose = (role) => { setRole(role); onSelect(role) }
  return (
    <div className="role-select">
      <div style={{ textAlign: 'center' }}>
        <div className="hero-badge" style={{ margin: '0 auto 18px' }}>
          <span className="dot" /> 智能企业知识库
        </div>
        <h1 className="page-title" style={{ fontSize: '2.4rem' }}>请选择你的角色</h1>
        <p className="muted" style={{ marginTop: 8 }}>不同角色可见的文档范围不同，可随时在顶部导航切换</p>
      </div>
      <div className="feature-grid" style={{ maxWidth: 560, width: '100%' }}>
        <div className="card card-interactive role-card" onClick={() => choose('employee')}>
          <div className="card-icon">👤</div>
          <h3>员工</h3>
          <p className="muted" style={{ fontSize: '0.9rem', marginTop: 8 }}>查阅企业制度、文化、培训等公开资料</p>
        </div>
        <div className="card card-interactive role-card" onClick={() => choose('manager')}>
          <div className="card-icon">💼</div>
          <h3>经理</h3>
          <p className="muted" style={{ fontSize: '0.9rem', marginTop: 8 }}>含经理专属资料，可维护文档与错误诊断</p>
        </div>
      </div>
    </div>
  )
}
