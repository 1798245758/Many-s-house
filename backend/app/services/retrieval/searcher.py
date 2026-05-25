import struct
import math
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk


def keyword_search(db: Session, query: str, top_k: int = 10) -> list[int]:
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
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def hybrid_search(db: Session, query_vec: list[float], query_text: str, top_k: int = 10) -> list[Chunk]:
    chunk_ids = keyword_search(db, query_text, top_k=top_k)
    if not chunk_ids:
        chunks = db.query(Chunk).order_by(Chunk.id.desc()).limit(top_k).all()
        return chunks

    chunks = db.query(Chunk).filter(Chunk.id.in_(chunk_ids)).all()
    id_order = {cid: i for i, cid in enumerate(chunk_ids)}
    chunks.sort(key=lambda c: id_order.get(c.id, 999))
    return chunks[:top_k]
