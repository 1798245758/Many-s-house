import json

from sqlalchemy import Column, Integer, String, Text, ForeignKey, func
from sqlalchemy.orm import relationship
from app.database import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    embedding = Column(String)  # JSON序列化的embedding向量
    metadata_json = Column(Text)  # 存储chunk级别的元数据
    created_at = Column(String, server_default=func.datetime('now'))

    document = relationship("Document")

    @property
    def meta(self) -> dict:
        """chunk 级元数据（页码/章节/类型），无则空 dict"""
        if not self.metadata_json:
            return {}
        try:
            return json.loads(self.metadata_json) or {}
        except (ValueError, TypeError):
            return {}

    @property
    def page(self) -> int | None:
        """chunk 所在页码（PDF 逐页切块时写入，其余文档为 None）"""
        return self.meta.get("page")

    @property
    def section(self) -> str | None:
        """chunk 所属章节标题（PDF 标题识别产出）"""
        return self.meta.get("section")

    @property
    def ctype(self) -> str | None:
        """chunk 类型：None=正文 / table=表格独立块"""
        return self.meta.get("type")
