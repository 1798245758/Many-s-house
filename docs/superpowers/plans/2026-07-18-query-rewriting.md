# Query Rewriting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现问题改写功能，将用户口语化/模糊的问题改写为更适合检索的标准化查询。

**Architecture:** 新增 `rewriter.py` 模块，调用 DeepSeek API 对用户问题进行改写优化。在 `generator.py` 中集成改写逻辑，作为 RAG 流程的第一步。

**Tech Stack:** Python, httpx, DeepSeek API, pytest

---

## File Structure

| 文件 | 职责 |
|------|------|
| `backend/app/services/retrieval/rewriter.py` | 问题改写模块 |
| `backend/app/services/retrieval/generator.py` | 修改：集成改写逻辑 |
| `backend/tests/test_rewriter.py` | 单元测试 |

---

## Task 1: 创建 rewriter.py 模块

**Files:**
- Create: `backend/app/services/retrieval/rewriter.py`

- [ ] **Step 1: 创建 rewriter.py 文件**

```python
from app.services.deepseek import DeepSeekClient


REWRITE_PROMPT = """你是一个查询优化专家。请将用户的问题改写为更适合知识库检索的形式。

改写规则：
1. 补充可能缺失的关键信息
2. 将口语化表达转为书面语
3. 提取核心关键词
4. 保持问题原意，不要改变语义
5. 如果问题已经很清晰，直接返回原问题

用户问题：{question}

改写后的查询："""


def rewrite_query(client: DeepSeekClient, question: str) -> str:
    """
    调用 DeepSeek API 改写用户问题
    
    Args:
        client: DeepSeek API 客户端
        question: 用户原始问题
        
    Returns:
        改写后的查询文本
    """
    if not question or not question.strip():
        return question
    
    prompt = REWRITE_PROMPT.format(question=question)
    response = client.chat(context="", question=prompt)
    return response.strip()
```

- [ ] **Step 2: 验证文件创建成功**

Run: `cat backend/app/services/retrieval/rewriter.py`

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/retrieval/rewriter.py
git commit -m "feat: add query rewriter module"
```

---

## Task 2: 创建 rewriter.py 单元测试

**Files:**
- Create: `backend/tests/test_rewriter.py`

- [ ] **Step 1: 创建测试文件**

```python
import pytest
from unittest.mock import patch, MagicMock
from app.services.retrieval.rewriter import rewrite_query
from app.services.deepseek import DeepSeekClient


@pytest.fixture
def mock_client():
    client = MagicMock(spec=DeepSeekClient)
    return client


def test_rewrite_returns_improved_query(mock_client):
    mock_client.chat.return_value = "RAG检索增强生成技术定义"
    
    result = rewrite_query(mock_client, "那个啥，就是讲人工智能的那个")
    
    assert result == "RAG检索增强生成技术定义"
    mock_client.chat.assert_called_once()


def test_rewrite_returns_original_on_empty(mock_client):
    result = rewrite_query(mock_client, "")
    assert result == ""
    mock_client.chat.assert_not_called()


def test_rewrite_returns_original_on_whitespace(mock_client):
    result = rewrite_query(mock_client, "   ")
    assert result == "   "
    mock_client.chat.assert_not_called()


def test_rewrite_strips_whitespace(mock_client):
    mock_client.chat.return_value = "  优化后的查询  "
    
    result = rewrite_query(mock_client, "原始问题")
    
    assert result == "优化后的查询"


def test_rewrite_passes_correct_prompt(mock_client):
    mock_client.chat.return_value = "改写结果"
    
    rewrite_query(mock_client, "测试问题")
    
    call_args = mock_client.chat.call_args
    assert "测试问题" in call_args.kwargs["question"]
    assert call_args.kwargs["context"] == ""
```

- [ ] **Step 2: 运行测试验证失败**

Run: `cd backend && python -m pytest tests/test_rewriter.py -v`

Expected: FAIL with "ModuleNotFoundError: No module named 'app.services.retrieval.rewriter'"

- [ ] **Step 3: 运行测试验证通过**

Run: `cd backend && python -m pytest tests/test_rewriter.py -v`

Expected: All 5 tests PASS

- [ ] **Step 4: Commit**

```bash
git add backend/tests/test_rewriter.py
git commit -m "test: add unit tests for query rewriter"
```

---

## Task 3: 修改 generator.py 集成改写逻辑

**Files:**
- Modify: `backend/app/services/retrieval/generator.py`

- [ ] **Step 1: 添加 import**

在 `generator.py` 第 5 行后添加：

```python
from app.services.retrieval.rewriter import rewrite_query
```

完整文件头部：

```python
import struct
from sqlalchemy.orm import Session
from app.services.deepseek import DeepSeekClient
from app.services.retrieval.searcher import hybrid_search
from app.services.retrieval.rewriter import rewrite_query
from app.services.embedding import embed_text, EMBEDDING_DIM
from app.schemas.query import SourceInfo, QueryResponse
```

- [ ] **Step 2: 修改 generate_answer 函数**

将原来的：

```python
def generate_answer(db: Session, query_text: str, client: DeepSeekClient) -> QueryResponse:
    query_vec = embed_query(query_text)
    chunks = hybrid_search(db, query_vec, query_text, top_k=10)
```

改为：

```python
def generate_answer(db: Session, query_text: str, client: DeepSeekClient, use_rewrite: bool = True) -> QueryResponse:
    # Step 1: 问题改写
    if use_rewrite:
        rewritten = rewrite_query(client, query_text)
    else:
        rewritten = query_text
    
    # Step 2: 向量化改写后的问题
    query_vec = embed_query(rewritten)
    
    # Step 3: 检索相关文档块
    chunks = hybrid_search(db, query_vec, rewritten, top_k=10)
```

- [ ] **Step 3: 验证修改结果**

Run: `cat backend/app/services/retrieval/generator.py`

- [ ] **Step 4: 运行现有测试确保不破坏**

Run: `cd backend && python -m pytest tests/ -v`

Expected: All existing tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/retrieval/generator.py
git commit -m "feat: integrate query rewriting into RAG pipeline"
```

---

## Task 4: 创建集成测试

**Files:**
- Create: `backend/tests/test_rewriter_integration.py`

- [ ] **Step 1: 创建集成测试文件**

```python
import pytest
from unittest.mock import patch, MagicMock
from app.services.retrieval.generator import generate_answer
from app.services.deepseek import DeepSeekClient
from app.models.document import Document
from app.models.chunk import Chunk
from app.services.embedding import embed_text


def test_generate_answer_with_rewrite(db_session):
    # 创建测试文档和分块
    doc = Document(filename="test.pdf", file_type="pdf", file_size=1000, status="ready")
    db_session.add(doc)
    db_session.commit()
    
    chunk = Chunk(
        document_id=doc.id,
        chunk_index=0,
        content="RAG是检索增强生成的缩写，结合了检索和生成两个步骤。",
        token_count=20,
        embedding=embed_text("RAG是检索增强生成")
    )
    db_session.add(chunk)
    db_session.commit()
    
    # Mock DeepSeek 客户端
    mock_client = MagicMock(spec=DeepSeekClient)
    mock_client.chat.side_effect = [
        "RAG检索增强生成技术",  # 改写结果
        "RAG是检索增强生成的缩写..."  # 最终回答
    ]
    
    result = generate_answer(db_session, "那个啥，就是讲人工智能的那个", mock_client, use_rewrite=True)
    
    assert result.answer == "RAG是检索增强生成的缩写..."
    assert len(result.sources) == 1
    assert mock_client.chat.call_count == 2


def test_generate_answer_without_rewrite(db_session):
    # 创建测试文档和分块
    doc = Document(filename="test.pdf", file_type="pdf", file_size=1000, status="ready")
    db_session.add(doc)
    db_session.commit()
    
    chunk = Chunk(
        document_id=doc.id,
        chunk_index=0,
        content="RAG是检索增强生成的缩写。",
        token_count=15,
        embedding=embed_text("RAG是检索增强生成")
    )
    db_session.add(chunk)
    db_session.commit()
    
    # Mock DeepSeek 客户端
    mock_client = MagicMock(spec=DeepSeekClient)
    mock_client.chat.return_value = "RAG是检索增强生成的缩写..."
    
    result = generate_answer(db_session, "RAG是什么", mock_client, use_rewrite=False)
    
    assert result.answer == "RAG是检索增强生成的缩写..."
    assert mock_client.chat.call_count == 1
```

- [ ] **Step 2: 运行集成测试**

Run: `cd backend && python -m pytest tests/test_rewriter_integration.py -v`

Expected: All tests PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_rewriter_integration.py
git commit -m "test: add integration tests for query rewriting"
```

---

## Task 5: 运行完整测试套件

- [ ] **Step 1: 运行所有测试**

Run: `cd backend && python -m pytest tests/ -v`

Expected: All tests PASS

- [ ] **Step 2: 最终 Commit（如有需要）**

```bash
git add -A
git commit -m "feat: complete query rewriting implementation"
```

---

## 验证清单

- [ ] `rewriter.py` 模块创建成功
- [ ] 单元测试全部通过
- [ ] 集成测试全部通过
- [ ] 现有测试未被破坏
- [ ] `generate_answer` 支持 `use_rewrite` 参数
- [ ] 问题改写调用 DeepSeek API
- [ ] 空问题处理正确
