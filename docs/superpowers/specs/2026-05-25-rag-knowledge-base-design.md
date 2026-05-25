# RAG 知识库 — 设计文档

> 项目：RAG 知识库  
> 状态：设计阶段（已确认）  
> 日期：2026-05-25

---

## 1. 概述

从零构建一个 RAG（检索增强生成）知识库系统。用户上传 PDF / TXT / Markdown 文档，系统自动完成文档解析、清洗、切分、向量化与索引构建（离线链路），并提供基于混合检索 + DeepSeek LLM 的智能问答（在线链路）。

### 技术栈

| 层级 | 技术 |
|------|------|
| 前端 | React 18 + Vite 5 |
| 后端 | Python + FastAPI |
| 数据库 | SQLite（含 FTS5 全文索引） |
| LLM | DeepSeek API（Chat + Embedding） |
| 架构 | 前后端分离，分层单体 |

---

## 2. 架构

采用**分层单体**架构：单一 FastAPI 应用，按离线/在线链路清晰分模块，SQLite 作为唯一数据存储。

### 项目结构

```
NRG/
├── frontend/                  # React + Vite
│   └── src/
│       ├── pages/             # 6 个页面
│       │   ├── Home/          # 主页 — 知识库概览
│       │   ├── Chat/          # 提问对话页
│       │   ├── Documents/     # 文档管理（上传/查看/删除）
│       │   ├── History/       # 历史记录
│       │   ├── Profile/       # 个人信息 + API Key + 主题
│       │   └── AboutData/     # 关于数据 + 隐私条款
│       ├── components/        # 复用组件
│       ├── api/               # API 调用封装
│       ├── contexts/          # ThemeContext 等
│       └── App.jsx
├── backend/                   # Python + FastAPI
│   └── app/
│       ├── routers/           # API 路由层
│       │   ├── documents.py   # POST upload / GET list / DELETE
│       │   ├── query.py       # POST query
│       │   ├── history.py     # GET list / DELETE
│       │   └── profile.py     # GET/PUT profile / GET/PUT settings
│       ├── services/          # 业务逻辑层
│       │   ├── ingestion/     # 离线链路
│       │   │   ├── extractor.py    # 文档解析（PDF/TXT/MD）
│       │   │   ├── cleaner.py      # 文档清洗
│       │   │   ├── chunker.py      # 语义切分
│       │   │   ├── embedder.py     # 向量化（DeepSeek API）
│       │   │   └── indexer.py      # FTS5 + 向量索引
│       │   └── retrieval/     # 在线链路
│       │       ├── embedder.py     # Query 向量化
│       │       ├── searcher.py     # 混合检索
│       │       └── generator.py    # LLM 生成
│       ├── models/            # SQLAlchemy ORM 模型
│       ├── schemas/           # Pydantic 请求/响应
│       └── main.py            # 应用入口
├── data/                      # SQLite 数据库文件（运行态）
├── uploads/                   # 用户上传文件存储（运行态）
└── docs/superpowers/specs/    # 设计文档
```

---

## 3. 数据库设计 (SQLite)

```sql
-- 文档表
documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,        -- 原始文件名
    file_type   TEXT NOT NULL,        -- pdf / txt / md
    file_size   INTEGER,              -- 字节数
    chunk_count INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'pending', -- pending / processing / ready / error
    created_at  TEXT DEFAULT (datetime('now'))
);

-- 文本块表
chunks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index  INTEGER NOT NULL,     -- 块序号
    content      TEXT NOT NULL,        -- 文本内容
    token_count  INTEGER,              -- token 估算
    embedding    BLOB,                 -- float32 序列化（可选，也可存 JSON）
    created_at   TEXT DEFAULT (datetime('now'))
);

-- 查询历史表
query_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query_text  TEXT NOT NULL,
    answer_text TEXT NOT NULL,
    sources     TEXT,                  -- JSON: 来源 chunk 信息
    created_at  TEXT DEFAULT (datetime('now'))
);

-- 用户表（单用户模式）
users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    nickname    TEXT DEFAULT '用户',
    email       TEXT,
    avatar      TEXT
);

-- 设置表（Key-Value）
settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
-- 预置 key: api_key（DeepSeek API Key）, theme（light/dark）
```

---

## 4. API 设计

### 统一响应格式

```json
{ "code": "SUCCESS", "message": "操作成功", "data": { ... } }
```

### 端点清单

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/documents/upload` | 上传文档（multipart/form-data） |
| GET | `/api/documents` | 文档列表 |
| DELETE | `/api/documents/{id}` | 删除文档及关联 chunks |
| POST | `/api/query` | 发起查询，返回 AI 回答 |
| GET | `/api/history` | 查询历史列表 |
| DELETE | `/api/history/{id}` | 删除单条历史 |
| GET | `/api/profile` | 获取个人信息 |
| PUT | `/api/profile` | 更新个人信息 |
| GET | `/api/settings` | 获取设置（API Key / 主题） |
| PUT | `/api/settings` | 更新设置 |

### 错误码

| code | 含义 |
|------|------|
| SUCCESS | 成功 |
| PARAM_ERROR | 参数错误 |
| NOT_FOUND | 资源不存在 |
| API_KEY_MISSING | 未配置 API Key |
| DOC_PROCESS_ERROR | 文档处理失败 |
| LLM_ERROR | LLM 调用失败 |
| SERVER_ERROR | 服务器内部错误 |

---

## 5. 数据流

### 5.1 离线链路（文档上传到索引就绪）

```
用户上传文件
  → 保存到 uploads/，写入 documents 表（status=pending）
  → 文档解析（extractor）
    - PDF  → pdfplumber
    - TXT  → 直接读取
    - MD   → 转纯文本
  → 文档清洗（cleaner）：去多余空行、特殊符号
  → 语义切分（chunker）：按段落/标题边界切分，每块 500-1000 token
  → 向量化（embedder）：调用 DeepSeek Embedding API
  → 索引构建（indexer）：写入 chunks 表 + FTS5 全文索引
  → 更新 documents.status = ready
```

### 5.2 在线链路（用户提问）

```
用户输入 Query
  → Query 向量化（调用 DeepSeek Embedding API）
  → 混合检索召回 Top-K=10
    - 向量检索：计算余弦相似度，筛选 ≥ 阈值
    - 关键词检索：SQLite FTS5 MATCH
    - 融合去重排序
  → 上下文拼接（取 Top-K chunk 的 content）
  → LLM 生成（调用 DeepSeek Chat API）
    - System Prompt: "你是一个知识库助手，仅基于提供的上下文回答问题..."
    - User Message: "上下文:\n{context}\n\n问题: {query}"
  → 保存到 query_history
  → 返回 { answer, sources }
```

---

## 6. 前端页面

| 路由 | 页面 | 功能 |
|------|------|------|
| `/` | 主页 | 欢迎信息 + 快速入口（上传文档 / 开始提问） |
| `/chat` | 对话 | 输入框 + 回答展示，显示引用来源 |
| `/documents` | 文档管理 | 文件上传、文档列表、删除 |
| `/history` | 历史记录 | 查询历史列表，点击可查看详情 |
| `/profile` | 个人信息 | 昵称/邮箱编辑 + API Key 配置 + 主题切换 |
| `/about` | 关于数据 | 数据来源说明 + 隐私声明 + 使用条款 |

### 状态覆盖

- **加载态**：Skeleton / Spinner
- **空状态**：各页面的"暂无数据"占位
- **错误态**：红色提示 + 重试按钮（message 直接来自后端）
- **成功态**：正常数据展示
- **边界**：长文本截断、分页

---

## 7. 错误处理

### 后端

- 统一异常处理中间件，所有异常转换为 `{ code, message }`
- 日志记录（print / logging）
- 不为前端暴露堆栈信息

### 前端

- 统一 API 拦截层，根据 `code` 决定行为
- `message` 直接展示给用户（中文）
- 网络错误单独处理："网络连接失败，请检查网络"

---

## 8. 测试策略

| 层级 | 工具 | 覆盖率目标 |
|------|------|-----------|
| 后端单元测试 | pytest | ≥ 85% |
| 后端集成测试 | pytest + TestClient + 测试 SQLite | ≥ 85% |
| 前端组件测试 | Vitest + React Testing Library | ≥ 85% |

### 核心测试场景

- 文档上传 → 解析 → 切片 → 索引全流程
- Query → 向量化 → 检索 → 生成全流程
- 所有 API 端点的正常/错误路径
- 空文档库查询的降级处理
- DeepSeek API Mock（避免真实调用）

### TDD 流程

RED（先写失败的测试）→ GREEN（最小实现使测试通过）→ REFACTOR（重构优化）

---

## 9. 依赖管理

### 前端

- 镜像：npm 淘宝镜像
- 核心依赖：react 18.x, react-dom 18.x, vite 5.x, react-router-dom 6.x
- 测试：vitest, @testing-library/react, @testing-library/jest-dom
- 锁定版本：package-lock.json

### 后端

- 镜像：pip 清华源
- 核心依赖：fastapi, uvicorn, sqlalchemy 2.x, pydantic 2.x, pdfplumber, httpx, python-multipart
- 测试：pytest, pytest-cov, httpx（TestClient 兼容）
- 锁定版本：requirements.txt（精确版本号）

---

## 10. 开发流程

```
brainstorming（本阶段 ✓）
  → writing-plans（计划阶段，写多阶段独立文件）
  → subagent-driven-development（并行开发）
  → test-driven-development（RED-GREEN-REFACTOR）
  → requesting-code-review（规格合规性 → 代码质量）
  → 交付
```

### 关键约束

1. **计划优先**：所有计划文件完成后，方可启动开发
2. **TDD 严格**：不得先写实现再补测试
3. **审查自动**：两阶段审查自动触发

---

## 附录 A：已确认的关键决策

| 决策项 | 选择 |
|--------|------|
| 数据来源 | 文件上传（PDF / TXT / MD） |
| LLM 服务 | DeepSeek API |
| 架构方案 | 分层单体（方案 A） |
| 个人信息 | 基本资料 + API Key + 主题切换 |
| 关于数据 | 数据说明 + 隐私声明 + 使用条款 |
| Chunking 策略 | 语义切分（按段落/标题边界） |
| 检索策略 | 混合检索（向量 + FTS5 关键词） |
