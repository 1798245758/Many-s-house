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
