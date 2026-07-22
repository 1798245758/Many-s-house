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
