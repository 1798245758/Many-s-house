from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session
from app.models.document import Document
from app.services.deepseek import DeepSeekClient
from app.services.ingestion.extractor import extract_text
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import semantic_chunk
from app.services.ingestion.embedder import vectorize_chunks
from app.services.ingestion.indexer import save_chunks

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

def _is_image(file_type: str) -> bool:
    return f".{file_type}" in IMAGE_EXTENSIONS

def ingest_document(db: Session, file_path: Path, filename: str, file_type: str, deepseek_client: Optional[DeepSeekClient]):
    doc = Document(filename=filename, file_type=file_type, file_size=file_path.stat().st_size, status="processing")
    db.add(doc)
    db.commit()
    db.refresh(doc)
    try:
        if _is_image(file_type):
            if deepseek_client:
                try:
                    raw_text = deepseek_client.describe_image(str(file_path))
                except Exception:
                    raw_text = f"[图片文件: {filename}]\n由于模型限制，未能自动识别图片内容，文件已保存。"
            else:
                raw_text = f"[图片文件: {filename}]\n未配置 API Key，无法识别图片内容，文件已保存。"
        else:
            raw_text = extract_text(file_path)
        cleaned = clean_text(raw_text)
        chunks = semantic_chunk(cleaned)
        embeddings = vectorize_chunks(chunks)
        save_chunks(db, doc.id, chunks, embeddings)
    except Exception as e:
        doc.status = "error"
        db.commit()
        raise e
    return doc
