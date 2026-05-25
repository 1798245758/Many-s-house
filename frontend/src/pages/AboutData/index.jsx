export default function AboutData() {
  return (
    <div>
      <h1 className="page-title">关于数据</h1>
      <div className="card">
        <h3>数据来源</h3>
        <p style={{marginTop:8,lineHeight:1.8}}>
          本系统的知识数据完全来源于用户上传的私人文档。系统不自动抓取、收集任何外部数据。
          所有上传的文档仅用于构建本地知识库，用于回答用户的查询问题。
        </p>
      </div>
      <div className="card">
        <h3>隐私声明</h3>
        <p style={{marginTop:8,lineHeight:1.8}}>
          1. 用户上传的文档存储在本服务器本地，不会上传到任何第三方服务。<br />
          2. 查询问题会通过 DeepSeek API 进行处理以生成回答。<br />
          3. 用户可随时删除已上传的文档，系统将一并删除相关的数据和索引。<br />
          4. 我们不会收集用户的个人信息用于任何商业目的。
        </p>
      </div>
      <div className="card">
        <h3>使用条款</h3>
        <p style={{marginTop:8,lineHeight:1.8}}>
          1. 用户应确保上传的文档内容合法合规。<br />
          2. 本系统提供的 AI 回答仅供参考，不构成任何专业建议。<br />
          3. 用户需自行配置 DeepSeek API Key 以使用问答功能。<br />
          4. 开发者不对因使用本系统而产生的任何损失承担责任。
        </p>
      </div>
    </div>
  )
}
