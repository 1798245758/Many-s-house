import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk
from app.services import vector_store

logger = logging.getLogger(__name__)

# 批量处理大小
BATCH_SIZE = 100


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
    except Exception as e:
        logger.warning(f"FTS搜索失败，回退到关键词搜索: {e}")

    # 使用分页查询避免一次性加载所有数据
    query_lower = query.lower()
    keywords = query_lower.split()
    scored = []
    
    # 分批查询
    offset = 0
    while True:
        chunks = db.query(Chunk).offset(offset).limit(BATCH_SIZE).all()
        if not chunks:
            break
            
        for c in chunks:
            content_lower = c.content.lower()
            score = 0
            for kw in keywords:
                count = content_lower.count(kw)
                if count > 0:
                    score += count * 10
                    # 注意：这里去掉冗余检查，因为count > 0已经意味着kw在content_lower中
            if score > 0:
                scored.append((c.id, score))
        
        offset += BATCH_SIZE
        # 如果已经获取足够的结果，可以提前退出
        if len(scored) >= top_k * 2:  # 获取更多结果以便排序
            break
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in scored[:top_k]]


def vector_search(db: Session, query_vec: list[float], top_k: int = 10) -> list[tuple[int, float]]:
    """向量相似度搜索（ChromaDB ANN检索）

    Args:
        db: 保留参数以兼容旧签名，实际不再使用
        query_vec: 查询向量
        top_k: 返回条数

    Returns:
        list[(chunk_id, 余弦相似度)]，按相似度降序
    """
    results = vector_store.query(query_vec, top_k=top_k)
    if not results:
        logger.info("ChromaDB向量检索无结果（向量库可能为空或故障）")
    return results


def hybrid_search(db: Session, query_vec: list[float], query_text: str, top_k: int = 10,
                  exclude_doc_ids: list[int] | None = None) -> list[Chunk]:
    """混合搜索：结合关键词搜索和向量搜索

    exclude_doc_ids：需排除的文档 ID（员工角色排除经理专属文档），
    在关键词/向量/无命中兜底三条路径统一生效。
    """
    excluded = set(exclude_doc_ids or [])

    # 获取关键词搜索结果
    keyword_ids = keyword_search(db, query_text, top_k=top_k)
    
    # 获取向量搜索结果
    vector_results = vector_search(db, query_vec, top_k=top_k)
    vector_ids = [cid for cid, _ in vector_results]
    
    # 合并结果，去重（保持顺序）
    all_ids = list(dict.fromkeys(keyword_ids + vector_ids))[:top_k]
    
    if not all_ids:
        chunks = db.query(Chunk).order_by(Chunk.id.desc()).limit(top_k * 3).all()
        if excluded:
            chunks = [c for c in chunks if c.document_id not in excluded][:top_k]
        return chunks
    
    chunks = db.query(Chunk).filter(Chunk.id.in_(all_ids)).all()
    if excluded:
        chunks = [c for c in chunks if c.document_id not in excluded]
    id_order = {cid: i for i, cid in enumerate(all_ids)}
    chunks.sort(key=lambda c: id_order.get(c.id, len(all_ids)))
    return chunks[:top_k]
