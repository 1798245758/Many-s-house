from pathlib import Path
from typing import Optional
import json
from sqlalchemy.orm import Session
from app.models.document import Document
from app.services.ingestion.extractor import extract_document
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import semantic_chunk
from app.services.ingestion.embedder import vectorize_chunks
from app.services.ingestion.indexer import save_chunks

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

def _is_image(file_type: str) -> bool:
    return f".{file_type}" in IMAGE_EXTENSIONS


def _split_table(md: str, max_tokens: int = 1000) -> list[str]:
    """超长表格按行拆段，每段保留表头两行，行列结构不被拍平"""
    if len(md) <= max_tokens:
        return [md]
    lines = md.splitlines()
    head, body = lines[:2], lines[2:]
    parts, cur = [], list(head)
    for ln in body:
        if sum(len(x) + 1 for x in cur) + len(ln) > max_tokens:
            parts.append("\n".join(cur))
            cur = list(head)
        cur.append(ln)
    if len(cur) > len(head):
        parts.append("\n".join(cur))
    return parts

def ingest_document(db: Session, file_path: Path, filename: str, file_type: str, deepseek_client=None, doc_id: int = None, visibility: str | None = None):
    if doc_id:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        doc.filename = filename
        doc.file_type = file_type
        doc.file_size = file_path.stat().st_size
        doc.status = "processing"
        doc.chunk_count = 0
        doc.metadata_json = None
        doc.structure_json = None
        if visibility is not None:
            doc.visibility = visibility
        db.commit()
    else:
        doc = Document(filename=filename, file_type=file_type, file_size=file_path.stat().st_size, status="processing")
        # 可见性随首次提交落库，避免处理窗口期内经理文档对员工可见可检索
        if visibility is not None:
            doc.visibility = visibility
        db.add(doc)
        db.commit()
        db.refresh(doc)
    try:
        if _is_image(file_type):
            raw_text = f"[图片文件: {filename}]  (可上传 XMind 等思维导图导出图片，系统已保存)"
            metadata = {}
            structure = {}
            page_content = []
        else:
            doc_info = extract_document(file_path)
            raw_text = doc_info["content"]
            metadata = doc_info["metadata"]
            structure = doc_info["structure"]
            page_content = doc_info.get("page_content") or []
            
            # 存储元数据和结构信息
            doc.metadata_json = json.dumps(metadata, ensure_ascii=False)
            doc.structure_json = json.dumps(structure, ensure_ascii=False)
            db.commit()
        
        if page_content:
            # 逐页切块：chunk 携带页码/章节路径（上下文增强：章节前缀随内容一同向量化）
            headings = (structure or {}).get("headings") or []
            chunks, metas = [], []
            section, hidx = None, 0
            for pno, ptext in enumerate(page_content, start=1):
                while hidx < len(headings) and headings[hidx]["page"] <= pno:
                    section = headings[hidx]["text"]
                    hidx += 1
                prefix = f"【{section}】\n" if section else ""
                for c in semantic_chunk(clean_text(ptext)):
                    chunks.append(prefix + c)
                    metas.append({"page": pno} | ({"section": section} if section else {}))
            # 表格独立成块：结构化 Markdown 不随正文切分拆散行列
            for tb in (structure or {}).get("tables") or []:
                for part in _split_table(tb["markdown"]):
                    chunks.append(part)
                    metas.append({"page": tb["page"], "type": "table"})
        else:
            chunks = semantic_chunk(clean_text(raw_text))
            metas = [None] * len(chunks)
        embeddings = vectorize_chunks(chunks)
        save_chunks(db, doc.id, chunks, embeddings, metas)
    except Exception as e:
        doc.status = "error"
        db.commit()
        raise e
    return doc
