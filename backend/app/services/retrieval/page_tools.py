"""按需回读工具：read_page 按页码回读原文、get_tables 获取表格块。

对应视频方法论"Agent 轻量工具组"——检索命中后不塞满上下文，
而是按 (document_id, page) 精准回读原文页 / 表格结构，支撑 citation 跳转与二次核验。
"""
from sqlalchemy.orm import Session
from app.models.chunk import Chunk


def read_page(db: Session, document_id: int, page: int) -> dict | None:
    """按 chunk_index 顺序重组指定页的正文原文；无该页正文时返回 None"""
    chunks = (db.query(Chunk).filter(Chunk.document_id == document_id)
              .order_by(Chunk.chunk_index).all())
    body = [c for c in chunks if c.page == page and c.ctype is None]
    if not body:
        return None
    return {"document_id": document_id, "page": page,
            "content": "\n".join(c.content for c in body),
            "chunk_ids": [c.id for c in body]}


def get_tables(db: Session, document_id: int, page: int | None = None) -> list[dict]:
    """获取文档的表格块（Markdown 行列结构），可按页码过滤"""
    chunks = (db.query(Chunk).filter(Chunk.document_id == document_id)
              .order_by(Chunk.chunk_index).all())
    tables = [c for c in chunks if c.ctype == "table"
              and (page is None or c.page == page)]
    return [{"chunk_id": c.id, "page": c.page, "markdown": c.content}
            for c in tables]
