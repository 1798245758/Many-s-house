from sqlalchemy.orm import Session
from sqlalchemy import text
from app.models.chunk import Chunk
from app.models.document import Document
import json
import struct
import traceback
import sys
from app.services.embedding import EMBEDDING_DIM
from app.services import vector_store


def build_fts_index(engine):
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    content, content='chunks', content_rowid='id'
                )
            """))
            conn.execute(text("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"))
            conn.commit()
    except Exception:
        print("WARNING: FTS5 index build failed, falling back to vector-only search", file=sys.stderr)
        traceback.print_exc()


def save_chunks(db: Session, document_id: int, chunks_content: list[str], embeddings: list[bytes],
                metas: list[dict | None] | None = None):
    chunks = []
    vecs = []
    for i, (content, emb) in enumerate(zip(chunks_content, embeddings)):
        vec = list(struct.unpack(f"{EMBEDDING_DIM}f", emb))
        meta = metas[i] if metas and i < len(metas) else None
        chunk = Chunk(
            document_id=document_id,
            chunk_index=i,
            content=content,
            token_count=len(content),
            metadata_json=json.dumps(meta, ensure_ascii=False) if meta else None,
        )
        db.add(chunk)
        chunks.append(chunk)
        vecs.append(vec)
    # 先flush获取自增id，再写入ChromaDB
    db.flush()
    chunk_ids = [c.id for c in chunks]
    if not vector_store.upsert_chunks(chunk_ids, vecs, document_id, list(range(len(chunks)))):
        # 向量库写入失败时回滚，保证SQLite与ChromaDB一致
        db.rollback()
        raise RuntimeError("向量写入ChromaDB失败")
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc:
        doc.chunk_count = len(chunks_content)
        doc.status = "ready"
    db.commit()
    build_fts_index(db.get_bind())
