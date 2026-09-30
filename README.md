# 企业问答助手 —— 企业级知识库 RAG 系统

基于 **FastAPI + LangChain/LangGraph + ChromaDB + React** 的企业知识库检索增强生成（RAG）问答系统。支持多格式文档入库、意图识别、多跳检索、证据校验、多轮记忆与检索错误诊断。

## 功能特性

- **离线入库链路**：文档上传 → 文本抽取（PDF/Word/PPT/Excel/Markdown，含 OCR 兜底）→ 分块 → BGE 向量化 → ChromaDB 索引
- **在线问答链路（六段式管道）**：意图识别网关 → 查询改写 → 多跳检索 → 证据校验 → 答案生成 → 诊断快照采集
- **意图识别**：两阶段 LLM 调用，区分闲聊/知识问答/信息不全（自动澄清）
- **多轮记忆**：短期记忆（LangGraph Checkpointer + sessionStorage）+ 长期记忆（SqliteStore）
- **异步任务框架**：LangGraph 状态图 + SqliteSaver 状态持久化 + SSE 进度推送
- **检索错误诊断 Agent**：Query 追踪、多轮诊断上下文管理
- **角色权限**：员工/经理双角色确定性门禁
- **金标评测**：黄金标准样本集评估检索与生成质量

## 技术栈

| 层 | 技术 |
|---|---|
| 后端 | FastAPI、SQLAlchemy、LangChain (LCEL)、LangGraph |
| 向量库 | ChromaDB（嵌入式） |
| Embedding | BAAI/bge-base-zh-v1.5（sentence-transformers） |
| LLM | DeepSeek（OpenAI 兼容接口） |
| 文档解析 | pdfplumber、pypdf、unstructured、docx2txt、python-pptx、openpyxl、pytesseract (OCR) |
| 前端 | React 18、Vite、React Router |

## 项目结构

```
├── backend/
│   ├── app/
│   │   ├── models/          # SQLAlchemy 模型（Document、Chunk、QueryTrace 等）
│   │   ├── routers/         # API 路由（documents、query、diagnosis、memory、tasks）
│   │   ├── schemas/         # Pydantic 请求/响应模型
│   │   ├── services/
│   │   │   ├── ingestion/   # 离线链路：extractor、chunker、embedder、indexer、pipeline
│   │   │   ├── retrieval/   # 在线链路：intent/rewrite/hop/verify/memory 链、agent_chain
│   │   │   ├── diagnosis/   # 检索错误诊断 Agent
│   │   │   ├── tasks/       # 异步任务框架（状态机 + CheckPoint）
│   │   │   ├── deepseek.py  # LLM 客户端
│   │   │   ├── embedding.py # BGE 向量化服务
│   │   │   └── ocr.py       # 扫描件 OCR 兜底
│   │   ├── config.py        # 配置管理
│   │   └── main.py          # FastAPI 入口
│   └── tests/               # pytest 测试集（含金标评测）
├── frontend/
│   └── src/
│       ├── pages/           # Home、Documents、Chat、Diagnosis、Memory、History 等
│       ├── components/      # Navbar、RoleSelect、Loading 等
│       ├── api/             # 后端接口封装
│       └── contexts/        # 主题上下文
├── docs/                    # 设计文档与规格说明
└── PDF_Example/             # 金标评测 PDF 样本（脚本可重新生成）
```

## 快速开始

### 后端

```bash
cd backend
pip install -r requirements.txt
copy .env.example .env       # 按需配置 LLM API Key 等
python -m uvicorn app.main:app --reload --port 8000
```

### 前端

```bash
cd frontend
npm install
npm run dev          # 同时拉起前后端
# 或 npm run dev:frontend  # 仅前端（默认 http://localhost:5173）
```

### 测试

```bash
cd backend
python -m pytest tests/ -v
```

## 使用流程

1. 在「文档管理」页上传企业文档（支持 PDF/DOCX/PPTX/XLSX/MD 等）
2. 系统自动完成抽取、分块、向量化与索引（异步任务 + SSE 进度）
3. 在「智能问答」页选择角色后提问，系统经意图识别、多跳检索、证据校验后生成带引用的回答
4. 支持多轮对话追问；「诊断」页可查看检索链路的诊断信息

## 说明

- 知识库文档（企业模拟文档）不随仓库分发
- 首次启动会自动初始化 SQLite 元数据库与 ChromaDB 向量库（`backend/data/`）
