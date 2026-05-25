import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div>
      <h1 className="page-title">RAG 知识库</h1>
      <p style={{marginBottom:24,color:'#666'}}>基于 DeepSeek 的智能知识问答系统，上传文档即可与你的私有知识对话。</p>
      <div style={{display:'flex',gap:16,flexWrap:'wrap'}}>
        <Link to="/documents" className="card" style={{flex:1,minWidth:200,textDecoration:'none',color:'inherit'}}>
          <h3>上传文档</h3>
          <p style={{color:'#666',marginTop:8}}>支持 PDF / TXT / Markdown</p>
        </Link>
        <Link to="/chat" className="card" style={{flex:1,minWidth:200,textDecoration:'none',color:'inherit'}}>
          <h3>开始提问</h3>
          <p style={{color:'#666',marginTop:8}}>基于私有文档智能问答</p>
        </Link>
      </div>
    </div>
  )
}
