"""混合检索模块（Hybrid Search）

职责：向量 ANN + 关键词 FTS5 混合检索，带角色权限过滤
使用方式：直接复制到项目 app/services/retrieval/searcher.py

关键设计：
- 双路召回：keyword_search（精确匹配）+ vector_search（语义召回）
- 合并策略：keyword 优先（精度高），vector 补充（召回广），dict.fromkeys 去重保序
- 权限过滤：exclude_doc_ids 在三条路径统一生效
- 无命中兜底：返回最新入库的 chunks（确保不空手而归）
- 分页扫描：FTS5 失败时退化为逐批遍历打分，BATCH_SIZE=100 控制内存
"""
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk
from app.services import vector_store

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def keyword_search(db: Session, query: str, top_k: int = 10) -> list[int]:
    """关键词检索（SQLite FTS5 全文索引）

    优先走 FTS5 MATCH；FTS 表不存在或查询语法错误时，
    退化为逐批遍历 + 词频打分。

    Returns:
        list[chunk_id]，按相关性降序
    """
    # 优先 FTS5
    try:
        result = db.execute(
            text("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH :q LIMIT :k"),
            {"q": query, "k": top_k},
        )
        ids = [row[0] for row in result.fetchall()]
        if ids:
            return ids
    except Exception as e:
        logger.warning(f"FTS搜索失败，回退到关键词遍历: {e}")

    # 退化：逐批遍历打分
    query_lower = query.lower()
    keywords = query_lower.split()
    scored = []

    offset = 0
    while True:
        chunks = db.query(Chunk).offset(offset).limit(BATCH_SIZE).all()
        if not chunks:
            break
        for c in chunks:
            content_lower = c.content.lower()
            score = sum(content_lower.count(kw) * 10
                        for kw in keywords if content_lower.count(kw) > 0)
            if score > 0:
                scored.append((c.id, score))
        offset += BATCH_SIZE
        if len(scored) >= top_k * 2:
            break

    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:top_k]]


def vector_search(db: Session, query_vec: list[float], top_k: int = 10) -> list[tuple[int, float]]:
    """向量 ANN 检索（委托 vector_store.query）

    Args:
        db: 保留参数兼容签名，实际不使用
        query_vec: 768 维查询向量
        top_k: 返回条数

    Returns:
        list[(chunk_id, cosine_similarity)]，按相似度降序
    """
    results = vector_store.query(query_vec, top_k=top_k)
    if not results:
        logger.info("向量检索无结果（向量库可能为空）")
    return results


def hybrid_search(db: Session, query_vec: list[float], query_text: str,
                  top_k: int = 10, exclude_doc_ids: list[int] | None = None) -> list[Chunk]:
    """混合检索：向量 + 关键词双路召回，带权限过滤

    这是 RAG 管道的核心检索入口。Agent 图中 search_node 应直接调用本函数。

    Args:
        db: SQLAlchemy Session（请求级，从 RunnableConfig 注入）
        query_vec: 查询向量（由 embed_node 生成）
        query_text: 查询文本（用于 FTS5 关键词匹配）
        top_k: 返回候选数
        exclude_doc_ids: 需排除的文档 ID 列表（角色权限过滤）

    Returns:
        list[Chunk]: ORM 对象列表，调用方需立即转为纯数据 dict

    合并策略：
        1. keyword_ids（FTS5 精确匹配，精度高）
        2. vector_ids（ANN 语义召回，覆盖广）
        3. dict.fromkeys 去重保序（keyword 优先占位）
        4. 两路均空 → 返回最新入库 chunks 兜底
    """
    excluded = set(exclude_doc_ids or [])

    # 双路召回
    keyword_ids = keyword_search(db, query_text, top_k=top_k)
    vector_results = vector_search(db, query_vec, top_k=top_k)
    vector_ids = [cid for cid, _ in vector_results]

    # 合并去重（keyword 优先）
    all_ids = list(dict.fromkeys(keyword_ids + vector_ids))[:top_k]

    # 无命中兜底：返回最新 chunks
    if not all_ids:
        chunks = db.query(Chunk).order_by(Chunk.id.desc()).limit(top_k * 3).all()
        if excluded:
            chunks = [c for c in chunks if c.document_id not in excluded][:top_k]
        return chunks

    # 按 ID 批量加载 + 权限过滤 + 保持合并顺序
    chunks = db.query(Chunk).filter(Chunk.id.in_(all_ids)).all()
    if excluded:
        chunks = [c for c in chunks if c.document_id not in excluded]
    id_order = {cid: i for i, cid in enumerate(all_ids)}
    chunks.sort(key=lambda c: id_order.get(c.id, len(all_ids)))
    return chunks[:top_k]
