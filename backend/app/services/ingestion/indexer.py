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
        conn.execute(text("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')"))
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
