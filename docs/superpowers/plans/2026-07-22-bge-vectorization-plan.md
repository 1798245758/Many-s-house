# BGE模型向量化集成实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目中的伪向量化实现替换为真正的向量化，使用国内开源的BGE模型（bge-base-zh-v1.5）。

**Architecture:** 使用sentence-transformers库加载BGE模型，修改embedding.py实现真正的文本嵌入，更新数据库维度，实现向量相似度搜索。

**Tech Stack:** Python, sentence-transformers, SQLAlchemy, FastAPI

---

### Task 1: 更新依赖和环境准备

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1: 添加sentence-transformers依赖**

在 `backend/requirements.txt` 文件末尾添加：

```python
sentence-transformers>=2.2.0
```

- [ ] **Step 2: 验证依赖安装**

Run: `cd backend && pip install -r requirements.txt`
Expected: 成功安装sentence-transformers及其依赖

- [ ] **Step 3: 提交更改**

```bash
git add backend/requirements.txt
git commit -m "deps: 添加sentence-transformers依赖用于BGE模型"
```

---

### Task 2: 修改embedding.py实现真正的向量化

**Files:**
- Modify: `backend/app/services/embedding.py`

- [ ] **Step 1: 读取当前embedding.py文件**

Run: `cat backend/app/services/embedding.py`
Expected: 查看当前的伪向量化实现

- [ ] **Step 2: 重写embedding.py**

```python
from sentence_transformers import SentenceTransformer
import struct
import numpy as np

# 全局模型实例，避免重复加载
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    return _model

EMBEDDING_DIM = 768


def embed_text(text: str) -> bytes:
    """将文本转换为BGE向量"""
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    # 转换为bytes格式
    return struct.pack(f"{EMBEDDING_DIM}f", *embedding.tolist())


def embed_text_list(texts: list[str]) -> list[bytes]:
    """批量将文本转换为BGE向量"""
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [struct.pack(f"{EMBEDDING_DIM}f", *emb.tolist()) for emb in embeddings]
```

- [ ] **Step 3: 测试embedding功能**

Run: `cd backend && python -c "from app.services.embedding import embed_text; print(len(embed_text('测试文本')))"`
Expected: 输出768（向量维度）

- [ ] **Step 4: 提交更改**

```bash
git add backend/app/services/embedding.py
git commit -m "feat: 使用BGE模型实现真正的文本向量化"
```

---

### Task 3: 更新ingestion/embedder.py

**Files:**
- Modify: `backend/app/services/ingestion/embedder.py`

- [ ] **Step 1: 读取当前embedder.py文件**

Run: `cat backend/app/services/ingestion/embedder.py`
Expected: 查看当前的向量化调用

- [ ] **Step 2: 重写embedder.py**

```python
from app.services.embedding import embed_text, embed_text_list


def vectorize_chunks(texts: list[str], client=None) -> list[bytes]:
    """将文本块转换为向量"""
    return embed_text_list(texts)
```

- [ ] **Step 3: 提交更改**

```bash
git add backend/app/services/ingestion/embedder.py
git commit -m "feat: 更新embedder使用新的BGE向量化"
```

---

### Task 4: 更新数据库维度常量

**Files:**
- Modify: `backend/app/services/ingestion/indexer.py`

- [ ] **Step 1: 读取当前indexer.py文件**

Run: `cat backend/app/services/ingestion/indexer.py`
Expected: 查看当前的向量存储逻辑

- [ ] **Step 2: 验证EMBEDDING_DIM导入**

确认第9行导入了 `EMBEDDING_DIM`，无需修改，因为embedding.py已经更新了维度。

- [ ] **Step 3: 测试向量存储**

Run: `cd backend && python -c "from app.services.embedding import EMBEDDING_DIM; print(f'向量维度: {EMBEDDING_DIM}')"`
Expected: 输出 `向量维度: 768`

- [ ] **Step 4: 提交更改**

```bash
git add backend/app/services/ingestion/indexer.py
git commit -m "chore: 验证向量维度更新"
```

---

### Task 5: 实现向量相似度搜索

**Files:**
- Modify: `backend/app/services/retrieval/searcher.py`

- [ ] **Step 1: 读取当前searcher.py文件**

Run: `cat backend/app/services/retrieval/searcher.py`
Expected: 查看当前的搜索实现

- [ ] **Step 2: 重写searcher.py实现混合搜索**

```python
import struct
import math
import json
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk
from app.services.embedding import embed_text, EMBEDDING_DIM


def keyword_search(db: Session, query: str, top_k: int = 10) -> list[int]:
    """关键词搜索"""
    try:
        result = db.execute(
            text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH :q LIMIT :k"),
            {"q": query, "k": top_k},
        )
        ids = [row[0] for row in result.fetchall()]
        if ids:
            return ids
    except Exception:
        pass

    chunks = db.query(Chunk).all()
    query_lower = query.lower()
    keywords = query_lower.split()
    scored = []
    for c in chunks:
        content_lower = c.content.lower()
        score = 0
        for kw in keywords:
            count = content_lower.count(kw)
            if count > 0:
                score += count * 10
                score += 5 if kw in content_lower else 0
        if score > 0:
            scored.append((c.id, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:top_k]]


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算余弦相似度"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search(db: Session, query_vec: list[float], top_k: int = 10) -> list[tuple[int, float]]:
    """向量相似度搜索"""
    chunks = db.query(Chunk).all()
    scored = []
    for c in chunks:
        if c.embedding:
            try:
                chunk_vec = json.loads(c.embedding)
                similarity = cosine_similarity(query_vec, chunk_vec)
                scored.append((c.id, similarity))
            except (json.JSONDecodeError, TypeError):
                continue
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hybrid_search(db: Session, query_vec: list[float], query_text: str, top_k: int = 10) -> list[Chunk]:
    """混合搜索：结合关键词搜索和向量搜索"""
    # 获取关键词搜索结果
    keyword_ids = keyword_search(db, query_text, top_k=top_k)
    
    # 获取向量搜索结果
    vector_results = vector_search(db, query_vec, top_k=top_k)
    vector_ids = [cid for cid, _ in vector_results]
    
    # 合并结果，去重
    all_ids = list(dict.fromkeys(keyword_ids + vector_ids))[:top_k]
    
    if not all_ids:
        chunks = db.query(Chunk).order_by(Chunk.id.desc()).limit(top_k).all()
        return chunks
    
    chunks = db.query(Chunk).filter(Chunk.id.in_(all_ids)).all()
    id_order = {cid: i for i, cid in enumerate(all_ids)}
    chunks.sort(key=lambda c: id_order.get(c.id, 999))
    return chunks[:top_k]
```

- [ ] **Step 3: 测试向量搜索功能**

Run: `cd backend && python -c "from app.services.retrieval.searcher import cosine_similarity; print(cosine_similarity([1,0,0], [1,0,0]))"`
Expected: 输出1.0

- [ ] **Step 4: 提交更改**

```bash
git add backend/app/services/retrieval/searcher.py
git commit -m "feat: 实现向量相似度搜索和混合搜索"
```

---

### Task 6: 更新搜索调用以使用向量

**Files:**
- Modify: `backend/app/services/retrieval/generator.py`

- [ ] **Step 1: 读取当前generator.py文件**

Run: `cat backend/app/services/retrieval/generator.py`
Expected: 查看当前的搜索调用逻辑

- [ ] **Step 2: 修改generator.py使用向量搜索**

在 `generate_answer` 函数中，修改搜索调用：

```python
from app.services.embedding import embed_text

# 在函数内部
query_vec = list(struct.unpack(f"{EMBEDDING_DIM}f", embed_text(query_text)))
results = hybrid_search(db, query_vec, query_text, top_k=top_k)
```

- [ ] **Step 3: 添加必要的导入**

```python
import struct
from app.services.embedding import EMBEDDING_DIM
```

- [ ] **Step 4: 提交更改**

```bash
git add backend/app/services/retrieval/generator.py
git commit -m "feat: 更新搜索调用使用向量搜索"
```

---

### Task 7: 端到端测试

**Files:**
- Create: `backend/test_bge_integration.py`

- [ ] **Step 1: 创建测试文件**

```python
import pytest
from app.services.embedding import embed_text, embed_text_list, EMBEDDING_DIM

def test_embedding_dimension():
    """测试向量维度"""
    result = embed_text("测试文本")
    assert len(result) == EMBEDDING_DIM * 4  # 4 bytes per float

def test_embedding_list():
    """测试批量向量化"""
    texts = ["文本1", "文本2", "文本3"]
    results = embed_text_list(texts)
    assert len(results) == 3
    for result in results:
        assert len(result) == EMBEDDING_DIM * 4

def test_embedding_normalization():
    """测试向量归一化"""
    result = embed_text("测试文本")
    import struct
    vector = list(struct.unpack(f"{EMBEDDING_DIM}f", result))
    norm = sum(v * v for v in vector) ** 0.5
    assert abs(norm - 1.0) < 0.01  # 归一化向量的模应接近1
```

- [ ] **Step 2: 运行测试**

Run: `cd backend && pytest test_bge_integration.py -v`
Expected: 所有测试通过

- [ ] **Step 3: 提交测试文件**

```bash
git add backend/test_bge_integration.py
git commit -m "test: 添加BGE集成测试"
```

---

### Task 8: 性能优化和缓存

**Files:**
- Modify: `backend/app/services/embedding.py`

- [ ] **Step 1: 添加模型缓存机制**

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> bytes:
    """缓存文本的embedding结果"""
    return embed_text(text)
```

- [ ] **Step 2: 更新embedder.py使用缓存**

```python
from app.services.embedding import get_cached_embedding

def vectorize_chunks(texts: list[str], client=None) -> list[bytes]:
    """将文本块转换为向量（带缓存）"""
    return [get_cached_embedding(t) for t in texts]
```

- [ ] **Step 3: 提交更改**

```bash
git add backend/app/services/embedding.py backend/app/services/ingestion/embedder.py
git commit -m "perf: 添加embedding缓存机制"
```

---

### Task 9: 错误处理和日志

**Files:**
- Modify: `backend/app/services/embedding.py`

- [ ] **Step 1: 添加错误处理**

```python
import logging

logger = logging.getLogger(__name__)

def embed_text(text: str) -> bytes:
    """将文本转换为BGE向量"""
    try:
        model = get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return struct.pack(f"{EMBEDDING_DIM}f", *embedding.tolist())
    except Exception as e:
        logger.error(f"向量化失败: {e}")
        # 返回零向量作为后备
        return struct.pack(f"{EMBEDDING_DIM}f", *[0.0] * EMBEDDING_DIM)
```

- [ ] **Step 2: 提交更改**

```bash
git add backend/app/services/embedding.py
git commit -m "fix: 添加embedding错误处理"
```

---

### Task 10: 文档更新

**Files:**
- Modify: `README.md` 或创建 `docs/BGE_USAGE.md`

- [ ] **Step 1: 创建使用文档**

```markdown
# BGE模型使用说明

## 安装依赖
```bash
pip install -r requirements.txt
```

## 首次使用
首次使用时会自动下载BGE模型（约400MB），请确保网络连接。

## 配置
模型默认使用 `BAAI/bge-base-zh-v1.5`，如需更换模型，请修改 `app/services/embedding.py` 中的模型名称。

## API使用
```python
from app.services.embedding import embed_text, embed_text_list

# 单文本向量化
vector = embed_text("你的文本")

# 批量向量化
vectors = embed_text_list(["文本1", "文本2"])
```
```

- [ ] **Step 2: 提交文档**

```bash
git add docs/BGE_USAGE.md
git commit -m "docs: 添加BGE模型使用文档"
```

---

## 验证清单

完成所有任务后，运行以下验证：

1. **单元测试**: `pytest test_bge_integration.py -v`
2. **集成测试**: `python -m pytest tests/ -v`
3. **手动测试**: 
   - 上传一个文档，验证向量化是否正常工作
   - 执行搜索，验证搜索结果质量
4. **性能检查**: 
   - 检查内存占用是否在可接受范围
   - 验证响应时间是否可接受

## 回滚方案

如果出现问题，可以执行：

```bash
git revert HEAD  # 回滚最后一次提交
# 或者
git reset --hard HEAD~10  # 回滚到集成前状态
```