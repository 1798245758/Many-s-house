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
