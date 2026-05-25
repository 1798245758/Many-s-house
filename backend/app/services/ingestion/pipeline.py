from pathlib import Path
from sqlalchemy.orm import Session
from app.models.document import Document
from app.services.deepseek import DeepSeekClient
from app.services.ingestion.extractor import extract_text
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import semantic_chunk
from app.services.ingestion.embedder import vectorize_chunks
from app.services.ingestion.indexer import save_chunks

def ingest_document(db: Session, file_path: Path, filename: str, file_type: str, deepseek_client: DeepSeekClient):
    doc = Document(filename=filename, file_type=file_type, file_size=file_path.stat().st_size, status="processing")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    try:
        raw_text = extract_text(file_path)
        cleaned = clean_text(raw_text)
        chunks = semantic_chunk(cleaned)
        embeddings = vectorize_chunks(chunks, deepseek_client)
        save_chunks(db, doc.id, chunks, embeddings)
    except Exception as e:
        doc.status = "error"
        db.commit()
        raise e
    return doc
