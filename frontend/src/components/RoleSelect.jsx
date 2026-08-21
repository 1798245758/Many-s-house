import { setRole } from '../api'

export default function RoleSelect({ onSelect }) {
  const choose = (role) => {
    setRole(role)
    onSelect(role)
  }
  return (
    <div style={{minHeight:'70vh',display:'flex',flexDirection:'column',justifyContent:'center',alignItems:'center',gap:24}}>
      <h1 className="page-title">请选择你的角色</h1>
      <p style={{color:'#666'}}>不同角色可见的文档范围不同，可随时在顶部导航切换</p>
      <div style={{display:'flex',gap:24,flexWrap:'wrap',justifyContent:'center'}}>
        <div className="card" style={{width:220,textAlign:'center',cursor:'pointer',padding:24}} onClick={() => choose('employee')}>
          <div style={{fontSize:'2.5rem'}}>👤</div>
          <h3>员工</h3>
          <p style={{color:'#666',fontSize:'0.9rem'}}>查阅企业制度、文化、培训等公开资料</p>
        </div>
        <div className="card" style={{width:220,textAlign:'center',cursor:'pointer',padding:24}} onClick={() => choose('manager')}>
          <div style={{fontSize:'2.5rem'}}>💼</div>
          <h3>经理</h3>
          <p style={{color:'#666',fontSize:'0.9rem'}}>含经理专属资料，可维护文档</p>
        </div>
      </div>
    </div>
  )
}
