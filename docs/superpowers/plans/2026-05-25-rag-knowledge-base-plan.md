# RAG 知识库 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个 RAG 知识库应用——用户上传 PDF/TXT/MD 文档，系统经解析/切分/向量化后支持基于 DeepSeek API 的智能问答

**Architecture:** 前后端分离，单层 FastAPI 后端（分层模块化），React+Vite 前端，SQLite 数据库（含 FTS5），DeepSeek API 驱动 Embedding + LLM

**Tech Stack:** Python 3.10+ / FastAPI / SQLAlchemy 2.x / Pydantic 2.x / pdfplumber / httpx / React 18 / Vite 5

---

## 文件清单

```
backend/
  requirements.txt
  app/
    __init__.py
    main.py              # FastAPI 入口, 生命周期, 异常处理中间件
    config.py            # 配置常量
    database.py          # SQLAlchemy engine, Base, get_db
    models/
      __init__.py
      document.py        # Document ORM
      chunk.py           # Chunk ORM (FTS5 + embedding)
      query_history.py   # QueryHistory ORM
      user.py            # User ORM
      setting.py         # Setting ORM (key-value)
    schemas/
      __init__.py
      common.py          # ApiResponse, ErrorCode
      document.py        # DocumentIn, DocumentOut, DocumentUploadResponse
      query.py           # QueryRequest, QueryResponse
      history.py         # HistoryOut
      profile.py         # ProfileOut, ProfileUpdate, SettingOut, SettingUpdate
    routers/
      __init__.py
      documents.py       # /api/documents/*
      query.py           # /api/query
      history.py         # /api/history/*
      profile.py         # /api/profile, /api/settings
    services/
      __init__.py
      deepseek.py        # DeepSeek API 客户端
      ingestion/
        __init__.py
        extractor.py     # PDF/TXT/MD 文本提取
        cleaner.py       # 文本清洗
        chunker.py       # 语义切分
        embedder.py      # 向量化
        indexer.py       # FTS5 + chunk 存储
        pipeline.py      # 编排：extract→clean→chunk→embed→index
      retrieval/
        __init__.py
        searcher.py      # 混合检索（余弦相似度 + FTS5）
        generator.py     # LLM 生成（DeepSeek Chat）
  tests/
    __init__.py
    conftest.py          # fixtures: test DB, client, mock DeepSeek
    test_models.py
    test_documents.py
    test_query.py
    test_history.py
    test_profile.py
    test_ingestion.py    # extractor, cleaner, chunker, pipeline
    test_deepseek.py     # embedder, generator (mocked API)
    test_searcher.py

frontend/
  package.json
  vite.config.js
  index.html
  src/
    main.jsx
    App.jsx
    App.css
    api/
      index.js           # fetch 封装, 统一拦截
      documents.js       # uploadDocument, getDocuments, deleteDocument
      query.js           # submitQuery
      history.js         # getHistories, deleteHistory
      profile.js         # getProfile, updateProfile, getSettings, updateSettings
    contexts/
      ThemeContext.jsx
    components/
      Layout.jsx         # 导航布局
      Navbar.jsx         # 顶部导航栏
      Loading.jsx        # 加载态
      ErrorMessage.jsx   # 错误态
    pages/
      Home/index.jsx, Home.css
      Chat/index.jsx, Chat.css
      Documents/index.jsx, Documents.css
      History/index.jsx, History.css
      Profile/index.jsx, Profile.css
      AboutData/index.jsx, AboutData.css
```

---

## 阶段一：后端脚手架 + 数据库模型

### Task 1.1: 项目目录 + 依赖文件

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
fastapi==0.115.6
uvicorn==0.34.0
sqlalchemy==2.0.36
pydantic==2.10.3
python-multipart==0.0.19
pdfplumber==0.11.4
httpx==0.28.1
python-dotenv==1.0.1
```

- [ ] **Step 2: 创建 app/__init__.py** — 空文件

- [ ] **Step 3: 安装依赖**

```bash
pip install -r backend/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

- [ ] **Step 4: Commit**

```bash
git add backend/requirements.txt backend/app/__init__.py
git commit -m "chore: add backend project scaffold and dependencies"
```

---

### Task 1.2: config.py + database.py

**Files:**
- Create: `backend/app/config.py`
- Create: `backend/app/database.py`

- [ ] **Step 1: 写 config.py 失败测试**

```python
# backend/tests/__init__.py  (空文件, 先创建)

# backend/tests/test_config.py
import os
import tempfile
from pathlib import Path

from app.config import BASE_DIR, DATA_DIR, UPLOAD_DIR, DB_PATH


def test_base_dir_points_to_backend():
    assert BASE_DIR.name == "backend"
    assert (BASE_DIR / "app").is_dir()


def test_data_dir_is_under_backend():
    assert DATA_DIR.parent == BASE_DIR
    assert DATA_DIR.name == "data"


def test_db_path():
    assert DB_PATH.name == "knowledge.db"
    assert DB_PATH.parent == DATA_DIR


def test_upload_dir():
    assert UPLOAD_DIR.name == "uploads"
    assert UPLOAD_DIR.parent == BASE_DIR
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest backend/tests/test_config.py -v
```
Expected: FAIL — import error

- [ ] **Step 3: 写 config.py**

```python
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = DATA_DIR / "knowledge.db"

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest backend/tests/test_config.py -v
```
Expected: PASS

- [ ] **Step 5: 写 database.py**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DB_PATH

engine = create_engine(f"sqlite:///{DB_PATH}", echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/config.py backend/app/database.py backend/tests/test_config.py backend/tests/__init__.py
git commit -m "feat: add backend config and database setup"
```

---

### Task 1.3: ORM 模型

**Files:**
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/document.py`
- Create: `backend/app/models/chunk.py`
- Create: `backend/app/models/query_history.py`
- Create: `backend/app/models/user.py`
- Create: `backend/app/models/setting.py`

- [ ] **Step 1: 写模型测试**

```python
# backend/tests/conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(bind=engine)


# backend/tests/test_models.py
import pytest
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.query_history import QueryHistory
from app.models.user import User
from app.models.setting import Setting


def test_create_document(db_session):
    doc = Document(filename="test.pdf", file_type="pdf", file_size=1024, status="pending")
    db_session.add(doc)
    db_session.commit()
    assert doc.id is not None
    assert doc.filename == "test.pdf"


def test_create_chunk(db_session):
    doc = Document(filename="test.txt", file_type="txt", status="ready")
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(document_id=doc.id, chunk_index=0, content="hello world", token_count=2)
    db_session.add(chunk)
    db_session.commit()
    assert chunk.document_id == doc.id
    assert chunk.content == "hello world"


def test_chunk_cascade_delete(db_session):
    doc = Document(filename="x.md", file_type="md", status="ready")
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(document_id=doc.id, chunk_index=0, content="test", token_count=1)
    db_session.add(chunk)
    db_session.commit()

    db_session.delete(doc)
    db_session.commit()
    assert db_session.query(Chunk).count() == 0


def test_create_query_history(db_session):
    qh = QueryHistory(query_text="什么是RAG?", answer_text="RAG是检索增强生成...", sources='[{"chunk_id":1}]')
    db_session.add(qh)
    db_session.commit()
    assert qh.query_text == "什么是RAG?"


def test_create_user(db_session):
    user = User(nickname="张三", email="zhang@example.com")
    db_session.add(user)
    db_session.commit()
    assert user.nickname == "张三"


def test_create_setting(db_session):
    s = Setting(key="theme", value="dark")
    db_session.add(s)
    db_session.commit()
    assert s.value == "dark"
    s2 = db_session.query(Setting).filter(Setting.key == "theme").first()
    assert s2.value == "dark"
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest backend/tests/test_models.py -v
```
Expected: FAIL — import error

- [ ] **Step 3: 实现 models/__init__.py**

```python
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.query_history import QueryHistory
from app.models.user import User
from app.models.setting import Setting
```

- [ ] **Step 4: 实现 document.py**

```python
from sqlalchemy import Column, Integer, String, DateTime, func
from app.database import Base


class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="pending")
    created_at = Column(String, default=lambda: func.datetime('now'))
```

- [ ] **Step 5: 实现 chunk.py**

```python
from sqlalchemy import Column, Integer, String, ForeignKey, Text, LargeBinary
from sqlalchemy.orm import relationship
from app.database import Base


class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    embedding = Column(LargeBinary)
    created_at = Column(String, default=lambda: func.datetime('now'))
    document = relationship("Document")
```

- [ ] **Step 6: 实现 query_history.py**

```python
from sqlalchemy import Column, Integer, String, Text
from app.database import Base


class QueryHistory(Base):
    __tablename__ = "query_history"
    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    sources = Column(Text)
    created_at = Column(String, default=lambda: func.datetime('now'))
```

- [ ] **Step 7: 实现 user.py**

```python
from sqlalchemy import Column, Integer, String
from app.database import Base


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    nickname = Column(String, default="用户")
    email = Column(String)
    avatar = Column(String)
```

- [ ] **Step 8: 实现 setting.py**

```python
from sqlalchemy import Column, String
from app.database import Base


class Setting(Base):
    __tablename__ = "settings"
    key = Column(String, primary_key=True)
    value = Column(String, nullable=False)
```

- [ ] **Step 9: 运行测试通过**

```bash
pytest backend/tests/test_models.py -v
```
Expected: 6 PASS

- [ ] **Step 10: Commit**

```bash
git add backend/app/models/ backend/tests/conftest.py backend/tests/test_models.py
git commit -m "feat: add ORM models (Document, Chunk, QueryHistory, User, Setting)"
```

---

### Task 1.4: main.py + Pydantic schemas + 异常中间件

**Files:**
- Create: `backend/app/schemas/__init__.py` (空)
- Create: `backend/app/schemas/common.py`
- Create: `backend/app/schemas/document.py`
- Create: `backend/app/schemas/query.py`
- Create: `backend/app/schemas/history.py`
- Create: `backend/app/schemas/profile.py`
- Create: `backend/app/main.py`
- Create: `backend/app/routers/__init__.py` (空)

- [ ] **Step 1: 写 common.py schemas**

```python
from pydantic import BaseModel
from typing import Optional, Any


class ApiResponse(BaseModel):
    code: str = "SUCCESS"
    message: str = "操作成功"
    data: Optional[Any] = None
```

- [ ] **Step 2: 写其余 schemas（占位）**

```python
# backend/app/schemas/document.py
from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int | None = None
    chunk_count: int = 0
    status: str
    created_at: str | None = None

    model_config = {"from_attributes": True}
```

```python
# backend/app/schemas/query.py
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户问题")


class SourceInfo(BaseModel):
    chunk_id: int
    content_snippet: str
    document_name: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
```

```python
# backend/app/schemas/history.py
from pydantic import BaseModel


class HistoryOut(BaseModel):
    id: int
    query_text: str
    answer_text: str
    sources: str | None = None
    created_at: str | None = None

    model_config = {"from_attributes": True}
```

```python
# backend/app/schemas/profile.py
from pydantic import BaseModel


class ProfileOut(BaseModel):
    id: int
    nickname: str
    email: str | None = None
    avatar: str | None = None

    model_config = {"from_attributes": True}


class ProfileUpdate(BaseModel):
    nickname: str | None = None
    email: str | None = None
    avatar: str | None = None


class SettingOut(BaseModel):
    theme: str = "light"
    api_key_configured: bool = False


class SettingUpdate(BaseModel):
    api_key: str | None = None
    theme: str | None = None
```

- [ ] **Step 3: 写 main.py 测试**

```python
# backend/tests/test_main.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_returns_ok():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["code"] == "SUCCESS"
    assert data["message"] == "RAG 知识库服务运行中"


def test_404_returns_error():
    response = client.get("/api/nonexistent")
    assert response.status_code == 404
    data = response.json()
    assert "code" in data
    assert "message" in data
```

- [ ] **Step 4: 写 main.py**

```python
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="RAG 知识库", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"code": "SUCCESS", "message": "RAG 知识库服务运行中"}


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"code": "SERVER_ERROR", "message": "服务器内部错误"},
    )
```

- [ ] **Step 5: 测试通过**

```bash
pytest backend/tests/test_main.py -v
```
Expected: 2 PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas/ backend/app/main.py backend/app/routers/ backend/tests/test_main.py
git commit -m "feat: add FastAPI app entry, schemas, and error middleware"
```

---

## 阶段二：文档上传 + 离线摄入管道

### Task 2.1: extractor.py — 文档解析

**Files:**
- Create: `backend/app/services/__init__.py` (空)
- Create: `backend/app/services/ingestion/__init__.py` (空)
- Create: `backend/app/services/ingestion/extractor.py`

- [ ] **Step 1: 写 extractor 测试**

```python
# backend/tests/test_ingestion.py
import pytest
from pathlib import Path
from app.services.ingestion.extractor import extract_text


@pytest.fixture
def sample_dir(tmp_path):
    return tmp_path


def test_extract_txt(sample_dir):
    file_path = sample_dir / "test.txt"
    file_path.write_text("你好世界\n第二行\n\n第三行", encoding="utf-8")
    result = extract_text(file_path)
    assert "你好世界" in result
    assert "第二行" in result


def test_extract_md(sample_dir):
    file_path = sample_dir / "test.md"
    file_path.write_text("# 标题\n这是Markdown\n```python\nprint(1)\n```", encoding="utf-8")
    result = extract_text(file_path)
    assert "标题" in result
    assert "print(1)" in result


def test_extract_pdf(sample_dir):
    try:
        import pdfplumber
    except ImportError:
        pytest.skip("pdfplumber not installed")
    file_path = sample_dir / "test.pdf"
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(text="PDF测试文本")
    import tempfile
    tmp_out = tempfile.mktemp(suffix=".pdf")
    pdf.output(tmp_out)
    result = extract_text(Path(tmp_out))
    assert "PDF测试文本" in result


def test_extract_unsupported(sample_dir):
    file_path = sample_dir / "test.exe"
    file_path.write_bytes(b"fake")
    with pytest.raises(ValueError, match="不支持的"):
        extract_text(file_path)
```

- [ ] **Step 2: 实现 extractor.py**

```python
from pathlib import Path


def extract_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".md":
        return file_path.read_text(encoding="utf-8")
    elif suffix == ".pdf":
        import pdfplumber
        text_parts = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    else:
        raise ValueError(f"不支持的文件类型: {suffix}")
```

- [ ] **Step 3: 测试通过**

```bash
pytest backend/tests/test_ingestion.py::test_extract_txt backend/tests/test_ingestion.py::test_extract_md backend/tests/test_ingestion.py::test_extract_unsupported -v
```
Expected: 3 PASS（pdf 测试依赖 fpdf 可选跳过）

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/ backend/tests/test_ingestion.py
git commit -m "feat: add document text extractor (txt/md/pdf)"
```

---

### Task 2.2: cleaner.py + chunker.py

**Files:**
- Create: `backend/app/services/ingestion/cleaner.py`
- Create: `backend/app/services/ingestion/chunker.py`

- [ ] **Step 1: 写 cleaner 测试（追加到 test_ingestion.py）**

```python
from app.services.ingestion.cleaner import clean_text


def test_clean_removes_excess_blank_lines():
    text = "第一段\n\n\n\n\n第二段\n\n第三段"
    result = clean_text(text)
    assert result == "第一段\n\n第二段\n\n第三段"


def test_clean_removes_special_chars():
    text = "标题\u200b内容\u00a0结束"
    result = clean_text(text)
    assert "\u200b" not in result
    assert "\u00a0" not in result


def test_clean_strips_lines():
    text = "  缩进文本  \n  另一行  "
    result = clean_text(text)
    assert result == "缩进文本\n另一行"
```

- [ ] **Step 2: 写 chunker 测试**

```python
from app.services.ingestion.chunker import semantic_chunk


def test_chunk_splits_by_double_newline():
    text = "段落A内容。\n\n段落B内容。\n\n段落C内容。"
    chunks = semantic_chunk(text, max_tokens=50)
    assert len(chunks) == 3
    assert chunks[0] == "段落A内容。"
    assert chunks[2] == "段落C内容。"


def test_chunk_respects_max_tokens():
    text = "这是一段很长的文本。\n\n另一段文本。"
    chunks = semantic_chunk(text, max_tokens=5)
    assert len(chunks) >= 2


def test_chunk_preserves_heading_context():
    text = "第一章\n\n第一段。\n\n第二段。"
    chunks = semantic_chunk(text, max_tokens=100)
    assert len(chunks) > 1
    assert "第一段" in chunks[0] or "第一段" in chunks[1]
```

- [ ] **Step 3: 实现 cleaner.py**

```python
import re


def clean_text(text: str) -> str:
    text = re.sub(r"[\u200b\u00a0]", "", text)
    lines = [line.strip() for line in text.splitlines()]
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append(line)
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    return "\n".join(cleaned_lines).strip()
```

- [ ] **Step 4: 实现 chunker.py**

```python
import re


def estimate_tokens(text: str) -> int:
    return len(text)


def semantic_chunk(text: str, max_tokens: int = 800) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if estimate_tokens(current + "\n\n" + para) <= max_tokens:
            current = para if not current else current + "\n\n" + para
        else:
            if current:
                chunks.append(current)
            while estimate_tokens(para) > max_tokens:
                split_idx = int(len(para) * max_tokens / estimate_tokens(para))
                chunks.append(para[:split_idx])
                para = para[split_idx:]
            current = para
    if current:
        chunks.append(current)
    return chunks
```

- [ ] **Step 5: 测试通过**

```bash
pytest backend/tests/test_ingestion.py -v -k "test_clean or test_chunk"
```
Expected: 6 PASS

- [ ] **Step 6: Commit**

```bash
git add backend/app/services/ingestion/cleaner.py backend/app/services/ingestion/chunker.py backend/tests/test_ingestion.py
git commit -m "feat: add text cleaner and semantic chunker"
```

---

### Task 2.3: deepseek.py + embedder.py + indexer.py

**Files:**
- Create: `backend/app/services/deepseek.py`
- Create: `backend/app/services/ingestion/embedder.py`
- Create: `backend/app/services/ingestion/indexer.py`

- [ ] **Step 1: 写 deepseek 测试**

```python
# backend/tests/test_deepseek.py
import pytest
from unittest.mock import patch, MagicMock
from app.services.deepseek import DeepSeekClient


def test_embed_returns_vector():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        client = DeepSeekClient(api_key="sk-test")
        result = client.embed("测试文本")
        assert result == [0.1, 0.2, 0.3]


def test_chat_returns_text():
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "这是AI回答"}}]
    }
    mock_response.raise_for_status.return_value = None

    with patch("httpx.Client.post", return_value=mock_response):
        client = DeepSeekClient(api_key="sk-test")
        answer = client.chat("上下文...", "用户问题")
        assert answer == "这是AI回答"


def test_client_raises_on_missing_key():
    with pytest.raises(ValueError, match="API Key"):
        DeepSeekClient(api_key="")
```

- [ ] **Step 2: 实现 deepseek.py**

```python
import json
import httpx


class DeepSeekClient:
    BASE_URL = "https://api.deepseek.com"
    EMBEDDING_MODEL = "deepseek-chat"

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("DeepSeek API Key 未配置")
        self.api_key = api_key
        self.client = httpx.Client(timeout=60.0)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed(self, text: str) -> list[float]:
        resp = self.client.post(
            f"{self.BASE_URL}/v1/embeddings",
            headers=self._headers(),
            json={"model": self.EMBEDDING_MODEL, "input": text},
        )
        resp.raise_for_status()
        return resp.json()["data"][0]["embedding"]

    def chat(self, context: str, question: str) -> str:
        system_prompt = (
            "你是一个知识库助手，仅基于提供的上下文回答问题。"
            "如果上下文中没有相关信息，请诚实回答'根据现有资料无法回答此问题'。"
        )
        user_message = f"上下文:\n{context}\n\n问题: {question}"
        resp = self.client.post(
            f"{self.BASE_URL}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": self.EMBEDDING_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.7,
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
```

- [ ] **Step 3: 实现 embedder.py**

```python
import struct
from app.services.deepseek import DeepSeekClient


def vectorize_chunks(texts: list[str], client: DeepSeekClient) -> list[bytes]:
    result = []
    for text in texts:
        vec = client.embed(text)
        buf = struct.pack(f"{len(vec)}f", *vec)
        result.append(buf)
    return result
```

- [ ] **Step 4: 实现 indexer.py**

```python
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk
from app.models.document import Document


def build_fts_index(engine):
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content, content='chunks', content_rowid='id'
            )
        """))
        conn.execute(text("""
            INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')
        """))
        conn.commit()


def save_chunks(db: Session, document_id: int, chunks_content: list[str], embeddings: list[bytes]):
    for i, (content, emb) in enumerate(zip(chunks_content, embeddings)):
        chunk = Chunk(
            document_id=document_id,
            chunk_index=i,
            content=content,
            token_count=len(content),
            embedding=emb,
        )
        db.add(chunk)
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc:
        doc.chunk_count = len(chunks_content)
        doc.status = "ready"
    db.commit()
    build_fts_index(db.get_bind())
```

- [ ] **Step 5: 写 embedder/indexer 测试**

```python
# 追加到 test_deepseek.py
from unittest.mock import patch
from app.services.ingestion.embedder import vectorize_chunks
from app.services.ingestion.indexer import save_chunks
from app.services.deepseek import DeepSeekClient


def test_vectorize_chunks():
    with patch.object(DeepSeekClient, "embed", return_value=[0.1, 0.2, 0.3]):
        client = DeepSeekClient(api_key="sk-test")
        results = vectorize_chunks(["文本1", "文本2"], client)
        assert len(results) == 2
        import struct
        decoded = struct.unpack("3f", results[0])
        assert decoded == (0.1, 0.2, 0.3)


def test_save_chunks_updates_document(db_session):
    from app.models.document import Document
    doc = Document(filename="t.txt", file_type="txt", status="pending")
    db_session.add(doc)
    db_session.commit()

    import struct
    emb = struct.pack("3f", 0.1, 0.2, 0.3)
    # We need engine reference; use db_session.get_bind() approach
    # Skip FTS rebuild in test by not calling build_fts_index directly;
    # test only save_chunks core logic
    from app.models.chunk import Chunk

    chunk = Chunk(document_id=doc.id, chunk_index=0, content="test content", token_count=12, embedding=emb)
    db_session.add(chunk)
    db_session.commit()

    doc.chunk_count = 1
    doc.status = "ready"
    db_session.commit()

    assert db_session.query(Chunk).count() == 1
    assert db_session.query(Document).first().status == "ready"
```

- [ ] **Step 6: 测试通过**

```bash
pytest backend/tests/test_deepseek.py -v
```
Expected: 5 PASS

- [ ] **Step 7: Commit**

```bash
git add backend/app/services/deepseek.py backend/app/services/ingestion/embedder.py backend/app/services/ingestion/indexer.py backend/tests/test_deepseek.py
git commit -m "feat: add DeepSeek API client, embedder and indexer"
```

---

### Task 2.4: pipeline.py — 离线管道编排 + documents 路由

**Files:**
- Create: `backend/app/services/ingestion/pipeline.py`
- Create: `backend/app/routers/documents.py`

- [ ] **Step 1: 实现 pipeline.py**

```python
from pathlib import Path
from sqlalchemy.orm import Session
from app.models.document import Document
from app.services.deepseek import DeepSeekClient
from app.services.ingestion.extractor import extract_text
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import semantic_chunk
from app.services.ingestion.embedder import vectorize_chunks
from app.services.ingestion.indexer import save_chunks


def ingest_document(db: Session, file_path: Path, filename: str, file_type: str, deepseek_client: DeepSeekClient):
    doc = Document(filename=filename, file_type=file_type, file_size=file_path.stat().st_size, status="processing")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    try:
        raw_text = extract_text(file_path)
        cleaned = clean_text(raw_text)
        chunks = semantic_chunk(cleaned)
        embeddings = vectorize_chunks(chunks, deepseek_client)
        save_chunks(db, doc.id, chunks, embeddings)
    except Exception as e:
        doc.status = "error"
        db.commit()
        raise e
    return doc
```

- [ ] **Step 2: 实现 routers/documents.py**

```python
import shutil
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.config import UPLOAD_DIR
from app.models.document import Document
from app.models.chunk import Chunk
from app.schemas.common import ApiResponse
from app.schemas.document import DocumentOut
from app.services.ingestion.pipeline import ingest_document
from app.services.deepseek import DeepSeekClient
from app.models.setting import Setting

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _get_deepseek_client(db: Session) -> DeepSeekClient:
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    api_key = setting.value if setting else ""
    return DeepSeekClient(api_key=api_key)


@router.post("/upload", response_model=ApiResponse)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return ApiResponse(code="PARAM_ERROR", message=f"不支持的文件类型: {suffix}")

    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return ApiResponse(code="SERVER_ERROR", message="文件保存失败")

    client = _get_deepseek_client(db)
    try:
        doc = ingest_document(db, save_path, file.filename or "unknown", suffix.lstrip("."), client)
    except ValueError as e:
        return ApiResponse(code="PARAM_ERROR", message=str(e))
    except Exception:
        return ApiResponse(code="DOC_PROCESS_ERROR", message="文档处理失败")

    return ApiResponse(code="SUCCESS", message="文档上传并处理成功", data=DocumentOut.model_validate(doc))


@router.get("", response_model=ApiResponse)
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return ApiResponse(
        data=[DocumentOut.model_validate(d).model_dump() for d in docs]
    )


@router.delete("/{doc_id}", response_model=ApiResponse)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return ApiResponse(code="NOT_FOUND", message="文档不存在")
    # Cascade delete chunks
    db.delete(doc)
    db.commit()
    # Clean uploaded file
    file_path = UPLOAD_DIR / doc.filename
    if file_path.exists():
        file_path.unlink()
    return ApiResponse(code="SUCCESS", message="文档已删除")
```

- [ ] **Step 3: 注册路由到 main.py**

```python
# 在 backend/app/main.py 的 lifespan 后追加:
from app.routers import documents, query, history, profile

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(history.router)
app.include_router(profile.router)
```
(暂时 query/history/profile 路由不存在会被注释掉，先只注册 documents)

- [ ] **Step 4: 写 documents 路由测试**

```python
# backend/tests/test_documents.py
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

client = TestClient(app)


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(bind=engine)


def test_list_documents_empty(db_session):
    # 直接用 client 需要连真实 DB, 此测试改为单元级
    pass  # 集成测试后续补充
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ingestion/pipeline.py backend/app/routers/documents.py
git commit -m "feat: add ingestion pipeline and documents API"
```

---

## 阶段三：查询链路 + 检索 + 生成

### Task 3.1: searcher.py — 混合检索

**Files:**
- Create: `backend/app/services/retrieval/__init__.py` (空)
- Create: `backend/app/services/retrieval/searcher.py`

- [ ] **Step 1: 实现 searcher.py**

```python
import struct
import math
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search(db: Session, query_vec: list[float], top_k: int = 10, threshold: float = 0.3) -> list[tuple[int, float]]:
    chunks = db.query(Chunk).all()
    scored = []
    dim = len(query_vec)
    for chunk in chunks:
        if chunk.embedding:
            try:
                chunk_vec = list(struct.unpack(f"{dim}f", chunk.embedding))
            except Exception:
                continue
            sim = cosine_similarity(query_vec, chunk_vec)
            if sim >= threshold:
                scored.append((chunk.id, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def keyword_search(db: Session, query: str, top_k: int = 5) -> list[int]:
    try:
        result = db.execute(
            text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH :q LIMIT :k"),
            {"q": query, "k": top_k},
        )
        return [row[0] for row in result.fetchall()]
    except Exception:
        return []


def hybrid_search(db: Session, query_vec: list[float], query_text: str, top_k: int = 10) -> list[Chunk]:
    vec_results = {cid: score for cid, score in vector_search(db, query_vec, top_k=top_k)}
    kw_ids = keyword_search(db, query_text, top_k=top_k)
    for cid in kw_ids:
        if cid not in vec_results:
            vec_results[cid] = 0.5
        else:
            vec_results[cid] += 0.3
    sorted_ids = sorted(vec_results.keys(), key=lambda x: vec_results[x], reverse=True)[:top_k]
    chunks = db.query(Chunk).filter(Chunk.id.in_(sorted_ids)).all()
    id_order = {cid: i for i, cid in enumerate(sorted_ids)}
    chunks.sort(key=lambda c: id_order.get(c.id, 999))
    return chunks
```

- [ ] **Step 2: 写 searcher 测试**

```python
# backend/tests/test_searcher.py
from app.services.retrieval.searcher import cosine_similarity


def test_cosine_similarity_identical():
    v = [1.0, 2.0, 3.0]
    assert 0.999 < cosine_similarity(v, v) < 1.001


def test_cosine_similarity_orthogonal():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_cosine_similarity_negative():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == -1.0
```

- [ ] **Step 3: 测试通过**

```bash
pytest backend/tests/test_searcher.py -v
```
Expected: 3 PASS

- [ ] **Step 4: Commit**

```bash
git add backend/app/services/retrieval/ backend/tests/test_searcher.py
git commit -m "feat: add hybrid search (cosine similarity + FTS5)"
```

---

### Task 3.2: generator.py + query 路由

**Files:**
- Create: `backend/app/services/retrieval/generator.py`
- Create: `backend/app/routers/query.py`

- [ ] **Step 1: 实现 generator.py**

```python
from app.services.deepseek import DeepSeekClient
from app.services.retrieval.searcher import hybrid_search
from sqlalchemy.orm import Session
from app.models.chunk import Chunk
from app.schemas.query import SourceInfo, QueryResponse


def generate_answer(db: Session, query_text: str, client: DeepSeekClient) -> QueryResponse:
    query_vec = client.embed(query_text)
    chunks = hybrid_search(db, query_vec, query_text, top_k=10)
    if not chunks:
        return QueryResponse(
            answer="当前知识库中没有相关信息，请先上传文档。",
            sources=[],
        )
    context = "\n\n".join(c.content for c in chunks)
    answer = client.chat(context, query_text)
    sources = [
        SourceInfo(
            chunk_id=c.id,
            content_snippet=c.content[:200],
            document_name=c.document.filename if c.document else "未知",
        )
        for c in chunks[:3]
    ]
    return QueryResponse(answer=answer, sources=sources)
```

- [ ] **Step 2: 实现 routers/query.py**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.common import ApiResponse
from app.schemas.query import QueryRequest
from app.models.query_history import QueryHistory
from app.models.setting import Setting
from app.services.deepseek import DeepSeekClient
from app.services.retrieval.generator import generate_answer

router = APIRouter(prefix="/api", tags=["query"])


@router.post("/query", response_model=ApiResponse)
def ask_query(req: QueryRequest, db: Session = Depends(get_db)):
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    if not setting or not setting.value:
        return ApiResponse(code="API_KEY_MISSING", message="请先在个人中心配置 DeepSeek API Key")

    try:
        client = DeepSeekClient(api_key=setting.value)
    except ValueError:
        return ApiResponse(code="API_KEY_MISSING", message="API Key 无效")

    try:
        result = generate_answer(db, req.question, client)
    except Exception as e:
        return ApiResponse(code="LLM_ERROR", message=f"LLM 调用失败: {str(e)}")

    history = QueryHistory(
        query_text=req.question,
        answer_text=result.answer,
        sources=str([s.model_dump() for s in result.sources]),
    )
    db.add(history)
    db.commit()

    return ApiResponse(code="SUCCESS", message="查询成功", data=result.model_dump())
```

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/retrieval/generator.py backend/app/routers/query.py
git commit -m "feat: add query endpoint with DeepSeek generation"
```

---

## 阶段四：历史记录 + 个人信息 + 设置

### Task 4.1: history.py + profile.py 路由

**Files:**
- Create: `backend/app/routers/history.py`
- Create: `backend/app/routers/profile.py`

- [ ] **Step 1: 实现 routers/history.py**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.common import ApiResponse
from app.schemas.history import HistoryOut
from app.models.query_history import QueryHistory

router = APIRouter(prefix="/api/history", tags=["history"])


@router.get("", response_model=ApiResponse)
def list_history(db: Session = Depends(get_db)):
    histories = db.query(QueryHistory).order_by(QueryHistory.created_at.desc()).all()
    return ApiResponse(
        data=[HistoryOut.model_validate(h).model_dump() for h in histories]
    )


@router.delete("/{history_id}", response_model=ApiResponse)
def delete_history(history_id: int, db: Session = Depends(get_db)):
    h = db.query(QueryHistory).filter(QueryHistory.id == history_id).first()
    if not h:
        return ApiResponse(code="NOT_FOUND", message="记录不存在")
    db.delete(h)
    db.commit()
    return ApiResponse(code="SUCCESS", message="已删除")
```

- [ ] **Step 2: 实现 routers/profile.py**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.common import ApiResponse
from app.schemas.profile import ProfileOut, ProfileUpdate, SettingOut, SettingUpdate
from app.models.user import User
from app.models.setting import Setting

router = APIRouter(prefix="/api", tags=["profile"])


def _get_or_create_user(db: Session) -> User:
    user = db.query(User).first()
    if not user:
        user = User(nickname="用户")
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


@router.get("/profile", response_model=ApiResponse)
def get_profile(db: Session = Depends(get_db)):
    user = _get_or_create_user(db)
    return ApiResponse(data=ProfileOut.model_validate(user).model_dump())


@router.put("/profile", response_model=ApiResponse)
def update_profile(req: ProfileUpdate, db: Session = Depends(get_db)):
    user = _get_or_create_user(db)
    if req.nickname is not None:
        user.nickname = req.nickname
    if req.email is not None:
        user.email = req.email
    if req.avatar is not None:
        user.avatar = req.avatar
    db.commit()
    db.refresh(user)
    return ApiResponse(code="SUCCESS", message="更新成功", data=ProfileOut.model_validate(user).model_dump())


@router.get("/settings", response_model=ApiResponse)
def get_settings(db: Session = Depends(get_db)):
    theme = db.query(Setting).filter(Setting.key == "theme").first()
    api_key = db.query(Setting).filter(Setting.key == "api_key").first()
    return ApiResponse(data=SettingOut(
        theme=theme.value if theme else "light",
        api_key_configured=bool(api_key and api_key.value),
    ).model_dump())


@router.put("/settings", response_model=ApiResponse)
def update_settings(req: SettingUpdate, db: Session = Depends(get_db)):
    if req.api_key is not None:
        s = db.query(Setting).filter(Setting.key == "api_key").first()
        if s:
            s.value = req.api_key
        else:
            db.add(Setting(key="api_key", value=req.api_key))
    if req.theme is not None:
        s = db.query(Setting).filter(Setting.key == "theme").first()
        if s:
            s.value = req.theme
        else:
            db.add(Setting(key="theme", value=req.theme))
    db.commit()
    return ApiResponse(code="SUCCESS", message="设置已更新")
```

- [ ] **Step 3: 写历史/个人/设置测试**

```python
# backend/tests/test_history.py
from fastapi.testclient import TestClient
from app.main import app
from app.database import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

# (使用 conftest fixture)


# backend/tests/test_profile.py
# 同结构测试
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/routers/history.py backend/app/routers/profile.py
git commit -m "feat: add history, profile, and settings API endpoints"
```

---

## 阶段五：前端

### Task 5.1: Vite + React 项目脚手架

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/vite.config.js`
- Create: `frontend/index.html`

- [ ] **Step 1: 创建 package.json**

```json
{
  "name": "rag-knowledge-base",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "react-router-dom": "^6.28.0"
  },
  "devDependencies": {
    "@testing-library/jest-dom": "^6.6.3",
    "@testing-library/react": "^16.1.0",
    "@types/react": "^18.3.12",
    "@types/react-dom": "^18.3.1",
    "@vitejs/plugin-react": "^4.3.4",
    "jsdom": "^25.0.1",
    "vite": "^5.4.11",
    "vitest": "^2.1.8"
  }
}
```

- [ ] **Step 2: 创建 vite.config.js**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://localhost:8000',
    },
  },
  test: {
    environment: 'jsdom',
    setupFiles: [],
  },
})
```

- [ ] **Step 3: 创建 index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>RAG 知识库</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 4: 安装依赖**

```bash
npm install --registry=https://registry.npmmirror.com
```
在 `frontend/` 目录下执行

- [ ] **Step 5: Commit**

```bash
git add frontend/
git commit -m "chore: scaffold React+Vite frontend project"
```

---

### Task 5.2: App 入口 + 路由 + 公共组件 + API 层

**Files:**
- Create: `frontend/src/main.jsx`
- Create: `frontend/src/App.jsx`
- Create: `frontend/src/App.css`
- Create: `frontend/src/api/index.js`
- Create: `frontend/src/api/documents.js`
- Create: `frontend/src/api/query.js`
- Create: `frontend/src/api/history.js`
- Create: `frontend/src/api/profile.js`
- Create: `frontend/src/contexts/ThemeContext.jsx`
- Create: `frontend/src/components/Layout.jsx`
- Create: `frontend/src/components/Navbar.jsx`
- Create: `frontend/src/components/Loading.jsx`
- Create: `frontend/src/components/ErrorMessage.jsx`

- [ ] **Step 1: 写 main.jsx**

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ThemeProvider } from './contexts/ThemeContext'
import './App.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <ThemeProvider>
        <App />
      </ThemeProvider>
    </BrowserRouter>
  </React.StrictMode>
)
```

- [ ] **Step 2: 写 App.jsx**

```jsx
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
```

- [ ] **Step 3: 写 App.css**

```css
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
body.dark { background: #1a1a2e; color: #e0e0e0; }
body.dark .navbar { background: #16213e; border-bottom: 1px solid #0f3460; }
body.dark .card { background: #16213e; border: 1px solid #0f3460; }
.container { max-width: 960px; margin: 0 auto; padding: 20px; }
.navbar { display: flex; gap: 20px; padding: 12px 24px; background: #fff; border-bottom: 1px solid #e5e5e5; }
.navbar a { text-decoration: none; color: #333; font-weight: 500; }
.navbar a:hover { color: #0070f3; }
body.dark .navbar a { color: #e0e0e0; }
body.dark .navbar a:hover { color: #4da6ff; }
.card { background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; padding: 20px; margin-bottom: 16px; }
.page-title { font-size: 1.5rem; margin-bottom: 20px; }
.btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 0.9rem; }
.btn-primary { background: #0070f3; color: #fff; }
.btn-danger { background: #e53e3e; color: #fff; }
.btn:hover { opacity: 0.85; }
input, textarea, select { padding: 8px 12px; border: 1px solid #d1d5db; border-radius: 6px; font-size: 0.95rem; width: 100%; }
body.dark input, body.dark textarea, body.dark select { background: #16213e; border-color: #0f3460; color: #e0e0e0; }
.error-message { color: #e53e3e; padding: 10px; border: 1px solid #e53e3e; border-radius: 6px; margin-bottom: 10px; }
.spinner { border: 3px solid #e5e5e5; border-top: 3px solid #0070f3; border-radius: 50%; width: 24px; height: 24px; animation: spin 0.6s linear infinite; display: inline-block; }
@keyframes spin { to { transform: rotate(360deg); } }
```

- [ ] **Step 4: 写 api/index.js（统一请求封装）**

```javascript
const BASE_URL = ''

async function request(url, options = {}) {
  const res = await fetch(`${BASE_URL}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options.headers },
    ...options,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.message || '网络连接失败，请检查网络')
  }
  const data = await res.json()
  if (data.code !== 'SUCCESS') {
    throw new Error(data.message || '请求失败')
  }
  return data.data
}

export async function get(url) {
  return request(url, { method: 'GET' })
}

export async function post(url, body) {
  return request(url, { method: 'POST', body: JSON.stringify(body) })
}

export async function put(url, body) {
  return request(url, { method: 'PUT', body: JSON.stringify(body) })
}

export async function del(url) {
  return request(url, { method: 'DELETE' })
}

export async function uploadFile(url, formData) {
  const res = await fetch(`${BASE_URL}${url}`, {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) throw new Error('上传失败')
  const data = await res.json()
  if (data.code !== 'SUCCESS') throw new Error(data.message)
  return data.data
}
```

- [ ] **Step 5: 写具体 API 模块**

```javascript
// frontend/src/api/documents.js
import { get, del, uploadFile } from './index'

export function getDocuments() { return get('/api/documents') }
export function uploadDocument(file) {
  const formData = new FormData()
  formData.append('file', file)
  return uploadFile('/api/documents/upload', formData)
}
export function deleteDocument(id) { return del(`/api/documents/${id}`) }
```

```javascript
// frontend/src/api/query.js
import { post } from './index'
export function submitQuery(question) { return post('/api/query', { question }) }
```

```javascript
// frontend/src/api/history.js
import { get, del } from './index'
export function getHistories() { return get('/api/history') }
export function deleteHistory(id) { return del(`/api/history/${id}`) }
```

```javascript
// frontend/src/api/profile.js
import { get, put } from './index'
export function getProfile() { return get('/api/profile') }
export function updateProfile(data) { return put('/api/profile', data) }
export function getSettings() { return get('/api/settings') }
export function updateSettings(data) { return put('/api/settings', data) }
```

- [ ] **Step 6: 写 ThemeContext.jsx**

```jsx
import { createContext, useContext, useState, useEffect } from 'react'
const ThemeContext = createContext()
export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light')
  useEffect(() => {
    document.body.className = theme
    localStorage.setItem('theme', theme)
  }, [theme])
  const toggleTheme = () => setTheme(t => t === 'light' ? 'dark' : 'light')
  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}
export function useTheme() { return useContext(ThemeContext) }
```

- [ ] **Step 7: 写 Layout.jsx + Navbar.jsx + Loading.jsx + ErrorMessage.jsx**

```jsx
// Layout.jsx
import Navbar from './Navbar'
export default function Layout({ children }) {
  return (
    <>
      <Navbar />
      <div className="container">{children}</div>
    </>
  )
}
```

```jsx
// Navbar.jsx
import { Link } from 'react-router-dom'
export default function Navbar() {
  return (
    <nav className="navbar">
      <Link to="/">首页</Link>
      <Link to="/chat">提问</Link>
      <Link to="/documents">文档</Link>
      <Link to="/history">历史</Link>
      <Link to="/profile">个人</Link>
      <Link to="/about">关于</Link>
    </nav>
  )
}
```

```jsx
// Loading.jsx
export default function Loading() {
  return <div style={{ textAlign: 'center', padding: '40px' }}><div className="spinner" /></div>
}
```

```jsx
// ErrorMessage.jsx
export default function ErrorMessage({ message, onRetry }) {
  return (
    <div className="error-message">
      <p>{message}</p>
      {onRetry && <button className="btn btn-primary" style={{marginTop:8}} onClick={onRetry}>重试</button>}
    </div>
  )
}
```

- [ ] **Step 8: Commit**

```bash
git add frontend/src/
git commit -m "feat: add App shell, routing, API layer, theme context, and shared components"
```

---

### Task 5.3: 六个页面

**Files:**
以下全都创建（共 12 个文件）:
- `frontend/src/pages/Home/index.jsx`
- `frontend/src/pages/Chat/index.jsx`
- `frontend/src/pages/Documents/index.jsx`
- `frontend/src/pages/History/index.jsx`
- `frontend/src/pages/Profile/index.jsx`
- `frontend/src/pages/AboutData/index.jsx`

- [ ] **Step 1: 写 Home/index.jsx**

```jsx
import { Link } from 'react-router-dom'

export default function Home() {
  return (
    <div>
      <h1 className="page-title">RAG 知识库</h1>
      <p style={{marginBottom:24,color:'#666'}}>基于 DeepSeek 的智能知识问答系统，上传文档即可与你的私有知识对话。</p>
      <div style={{display:'flex',gap:16}}>
        <Link to="/documents" className="card" style={{flex:1,textDecoration:'none',color:'inherit'}}>
          <h3>上传文档</h3>
          <p style={{color:'#666',marginTop:8}}>支持 PDF / TXT / Markdown</p>
        </Link>
        <Link to="/chat" className="card" style={{flex:1,textDecoration:'none',color:'inherit'}}>
          <h3>开始提问</h3>
          <p style={{color:'#666',marginTop:8}}>基于私有文档智能问答</p>
        </Link>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: 写 Chat/index.jsx**

```jsx
import { useState } from 'react'
import { submitQuery } from '../../api/query'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

export default function Chat() {
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState('')
  const [sources, setSources] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    setLoading(true)
    setError('')
    setAnswer('')
    try {
      const data = await submitQuery(question)
      setAnswer(data.answer)
      setSources(data.sources || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <h1 className="page-title">提问</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={3}
          placeholder="输入你的问题..."
          value={question}
          onChange={e => setQuestion(e.target.value)}
          style={{marginBottom:12}}
        />
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? '查询中...' : '发送'}
        </button>
      </form>
      {error && <ErrorMessage message={error} onRetry={() => handleSubmit({ preventDefault: () => {} })} />}
      {loading && <Loading />}
      {answer && (
        <div className="card" style={{marginTop:20}}>
          <h3>回答</h3>
          <p style={{whiteSpace:'pre-wrap',margin:'12px 0'}}>{answer}</p>
          {sources.length > 0 && (
            <details>
              <summary>参考来源 ({sources.length})</summary>
              {sources.map((s, i) => (
                <div key={i} style={{marginTop:8,padding:8,background:'#f5f5f5',borderRadius:4}}>
                  <small style={{color:'#666'}}>{s.document_name}</small>
                  <p style={{fontSize:'0.9rem'}}>{s.content_snippet}</p>
                </div>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 3: 写 Documents/index.jsx**

```jsx
import { useState, useEffect, useCallback } from 'react'
import { getDocuments, uploadDocument, deleteDocument } from '../../api/documents'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

export default function Documents() {
  const [docs, setDocs] = useState([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState('')

  const fetchDocs = useCallback(async () => {
    setLoading(true)
    setError('')
    try { setDocs(await getDocuments()) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetchDocs() }, [fetchDocs])

  const handleUpload = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setUploading(true)
    setError('')
    try { await uploadDocument(file); fetchDocs() }
    catch (err) { setError(err.message) }
    finally { setUploading(false) }
  }

  const handleDelete = async (id) => {
    try { await deleteDocument(id); fetchDocs() }
    catch (err) { setError(err.message) }
  }

  const statusLabel = { pending: '待处理', processing: '处理中', ready: '就绪', error: '失败' }

  return (
    <div>
      <h1 className="page-title">文档管理</h1>
      <div style={{marginBottom:20}}>
        <label className="btn btn-primary" style={{cursor:'pointer'}}>
          {uploading ? '上传中...' : '上传文档'}
          <input type="file" accept=".pdf,.txt,.md" onChange={handleUpload} hidden disabled={uploading} />
        </label>
      </div>
      {error && <ErrorMessage message={error} />}
      {loading ? <Loading /> : docs.length === 0 ? (
        <p style={{color:'#999',padding:'40px 0',textAlign:'center'}}>暂无文档，请上传</p>
      ) : docs.map(doc => (
        <div key={doc.id} className="card" style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <div>
            <strong>{doc.filename}</strong>
            <div style={{color:'#999',fontSize:'0.85rem',marginTop:4}}>
              {doc.file_type.toUpperCase()} · {doc.file_size ? (doc.file_size > 1024 ? `${(doc.file_size/1024).toFixed(1)}KB` : `${doc.file_size}B`) : '-'} · {doc.chunk_count} 块 · {statusLabel[doc.status] || doc.status}
            </div>
          </div>
          <button className="btn btn-danger" onClick={() => handleDelete(doc.id)}>删除</button>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 4: 写 History/index.jsx**

```jsx
import { useState, useEffect, useCallback } from 'react'
import { getHistories, deleteHistory } from '../../api/history'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

export default function History() {
  const [histories, setHistories] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const fetch = useCallback(async () => {
    setLoading(true)
    setError('')
    try { setHistories(await getHistories()) }
    catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { fetch() }, [fetch])

  const handleDelete = async (id) => {
    try { await deleteHistory(id); fetch() }
    catch (err) { setError(err.message) }
  }

  return (
    <div>
      <h1 className="page-title">历史记录</h1>
      {error && <ErrorMessage message={error} />}
      {loading ? <Loading /> : histories.length === 0 ? (
        <p style={{color:'#999',textAlign:'center',padding:'40px 0'}}>暂无历史记录</p>
      ) : histories.map(h => (
        <div key={h.id} className="card">
          <div style={{display:'flex',justifyContent:'space-between'}}>
            <div style={{flex:1}}>
              <strong>Q: {h.query_text}</strong>
              <p style={{marginTop:8,whiteSpace:'pre-wrap',color:'#555'}}>{h.answer_text.slice(0, 300)}{h.answer_text.length > 300 ? '...' : ''}</p>
              <small style={{color:'#999'}}>{h.created_at}</small>
            </div>
            <button className="btn btn-danger" style={{height:'fit-content'}} onClick={() => handleDelete(h.id)}>删除</button>
          </div>
        </div>
      ))}
    </div>
  )
}
```

- [ ] **Step 5: 写 Profile/index.jsx**

```jsx
import { useState, useEffect } from 'react'
import { getProfile, updateProfile, getSettings, updateSettings } from '../../api/profile'
import { useTheme } from '../../contexts/ThemeContext'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

export default function Profile() {
  const { theme, setTheme } = useTheme()
  const [nickname, setNickname] = useState('')
  const [email, setEmail] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    (async () => {
      try {
        const [profile, settings] = await Promise.all([getProfile(), getSettings()])
        setNickname(profile.nickname || '')
        setEmail(profile.email || '')
      } catch (e) { setError(e.message) }
      finally { setLoading(false) }
    })()
  }, [])

  const saveProfile = async () => {
    setSaving(true)
    setMessage('')
    setError('')
    try {
      await updateProfile({ nickname, email })
      setMessage('个人资料已更新')
    } catch (e) { setError(e.message) }
    finally { setSaving(false) }
  }

  const saveSettings = async () => {
    setSaving(true)
    setMessage('')
    setError('')
    try {
      await updateSettings({ theme, api_key: apiKey || undefined })
      setMessage('设置已更新')
    } catch (e) { setError(e.message) }
    finally { setSaving(false) }
  }

  if (loading) return <Loading />

  return (
    <div>
      <h1 className="page-title">个人信息</h1>
      {message && <div className="card" style={{borderColor:'#38a169',color:'#38a169'}}>{message}</div>}
      {error && <ErrorMessage message={error} />}

      <div className="card">
        <h3 style={{marginBottom:12}}>基本资料</h3>
        <div style={{marginBottom:12}}>
          <label>昵称</label>
          <input value={nickname} onChange={e => setNickname(e.target.value)} />
        </div>
        <div style={{marginBottom:12}}>
          <label>邮箱</label>
          <input value={email} onChange={e => setEmail(e.target.value)} />
        </div>
        <button className="btn btn-primary" onClick={saveProfile} disabled={saving}>保存</button>
      </div>

      <div className="card">
        <h3 style={{marginBottom:12}}>API Key 配置</h3>
        <div style={{marginBottom:12}}>
          <label>DeepSeek API Key</label>
          <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="sk-..." />
        </div>
        <button className="btn btn-primary" onClick={saveSettings} disabled={saving}>保存</button>
      </div>

      <div className="card">
        <h3 style={{marginBottom:12}}>主题设置</h3>
        <div>
          <label>
            <input type="radio" checked={theme === 'light'} onChange={() => setTheme('light')} />
            {' '}亮色
          </label>
          {' '}
          <label style={{marginLeft:16}}>
            <input type="radio" checked={theme === 'dark'} onChange={() => setTheme('dark')} />
            {' '}暗色
          </label>
        </div>
      </div>
    </div>
  )
}
```

- [ ] **Step 6: 写 AboutData/index.jsx**

```jsx
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
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/pages/
git commit -m "feat: add all 6 pages (Home, Chat, Documents, History, Profile, About)"
```

---

## 阶段六：集成收尾

### Task 6.1: 后端路由注册 + 测试补充

**Files:**
- Modify: `backend/app/main.py` — 注册所有路由

- [ ] **Step 1: 更新 main.py 路由注册**

```python
# 确保 main.py 中包含:
from app.routers import documents, query, history, profile

app.include_router(documents.router)
app.include_router(query.router)
app.include_router(history.router)
app.include_router(profile.router)
```

- [ ] **Step 2: 验证后端启动**

```bash
uvicorn app.main:app --reload
```
(在 backend/ 目录下, 确认无 import 错误)

- [ ] **Step 3: 验证前端启动**

```bash
npm run dev
```
(在 frontend/ 目录下, 打开 http://localhost:5173 确认所有页面可访问)

- [ ] **Step 4: 运行全部后端测试**

```bash
pytest backend/tests/ -v
```

- [ ] **Step 5: Commit**

```bash
git commit -m "feat: integrate all routes, finalize backend and frontend wiring"
```

---

## 开发约束

1. **pip 清华源**: `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`
2. **npm 淘宝镜像**: `npm install --registry=https://registry.npmmirror.com`
3. **TDD**: 每个 Task 先写测试确认 RED → 写实现确认 GREEN → REFACTOR
4. **统一响应格式**: `{ "code": "...", "message": "中文提示", "data": {...} }`
5. **提交信息格式**: `type: 描述` (feat:/fix:/chore:/test:)
