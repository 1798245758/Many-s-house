from sqlalchemy import Column, Integer, String, ForeignKey, Text, LargeBinary, func
from sqlalchemy.orm import relationship
from app.database import Base


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    token_count = Column(Integer)
    embedding = Column(LargeBinary)
    created_at = Column(String, server_default=func.datetime('now'))

    document = relationship("Document")
