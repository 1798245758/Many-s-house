import math
import json
import logging
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk

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


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """计算余弦相似度"""
    # 验证向量维度
    if len(vec_a) != len(vec_b):
        logger.warning(f"向量维度不匹配: {len(vec_a)} vs {len(vec_b)}")
        return 0.0
    
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def vector_search(db: Session, query_vec: list[float], top_k: int = 10) -> list[tuple[int, float]]:
    """向量相似度搜索（分批处理）"""
    scored = []
    
    # 分批查询避免内存问题
    offset = 0
    while True:
        chunks = db.query(Chunk).offset(offset).limit(BATCH_SIZE).all()
        if not chunks:
            break
            
        for c in chunks:
            if c.embedding:
                try:
                    chunk_vec = json.loads(c.embedding)
                    similarity = cosine_similarity(query_vec, chunk_vec)
                    if similarity > 0:  # 只保留正相似度的结果
                        scored.append((c.id, similarity))
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"解析向量失败 (chunk_id={c.id}): {e}")
                    continue
        
        offset += BATCH_SIZE
        # 如果已经获取足够的结果，可以提前退出
        if len(scored) >= top_k * 2:  # 获取更多结果以便排序
            break
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def hybrid_search(db: Session, query_vec: list[float], query_text: str, top_k: int = 10) -> list[Chunk]:
    """混合搜索：结合关键词搜索和向量搜索"""
    # 获取关键词搜索结果
    keyword_ids = keyword_search(db, query_text, top_k=top_k)
    
    # 获取向量搜索结果
    vector_results = vector_search(db, query_vec, top_k=top_k)
    vector_ids = [cid for cid, _ in vector_results]
    
    # 合并结果，去重（保持顺序）
    all_ids = list(dict.fromkeys(keyword_ids + vector_ids))[:top_k]
    
    if not all_ids:
        chunks = db.query(Chunk).order_by(Chunk.id.desc()).limit(top_k).all()
        return chunks
    
    chunks = db.query(Chunk).filter(Chunk.id.in_(all_ids)).all()
    id_order = {cid: i for i, cid in enumerate(all_ids)}
    chunks.sort(key=lambda c: id_order.get(c.id, len(all_ids)))
    return chunks[:top_k]
