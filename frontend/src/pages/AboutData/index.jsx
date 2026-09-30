const SECTIONS = [
  {
    icon: '📊', title: '数据来源',
    body: <>本系统的知识数据来源于两部分：一是内置的企业管理类文档（覆盖企业文化、员工手册、管理制度、培训体系等主题），二是用户上传的私有文档。系统不自动抓取、收集其他外部数据。所有数据仅用于构建本地企业知识库，用于回答用户的查询问题。</>,
  },
  {
    icon: '🔒', title: '隐私声明',
    body: <>1. 用户上传的文档存储在本服务器本地，不会上传到任何第三方服务。<br />2. 查询问题会通过 DeepSeek API 进行处理以生成回答。<br />3. 用户可随时删除已上传的文档，系统将一并删除相关的数据和索引。<br />4. 我们不会收集用户的个人信息用于任何商业目的。</>,
  },
  {
    icon: '📜', title: '使用条款',
    body: <>1. 用户应确保上传的文档内容合法合规。<br />2. 本系统提供的回答仅供参考，具体企业管理决策请以公司正式发布的制度文件为准。<br />3. 用户需自行配置 DeepSeek API Key 以使用问答功能。<br />4. 开发者不对因使用本系统而产生的任何损失承担责任。</>,
  },
]

export default function AboutData() {
  return (
    <div style={{ maxWidth: 760, margin: '0 auto' }}>
      <h1 className="page-title">关于数据</h1>
      <p className="page-subtitle">数据来源、隐私声明与使用条款</p>
      <div className="stack stagger">
        {SECTIONS.map(s => (
          <div key={s.title} className="card card-hover" style={{ marginBottom: 0 }}>
            <h3 style={{ display: 'flex', alignItems: 'center', gap: 8 }}>{s.icon} {s.title}</h3>
            <p className="muted" style={{ marginTop: 10, lineHeight: 1.85 }}>{s.body}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
