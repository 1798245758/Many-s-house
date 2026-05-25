from sqlalchemy import Column, Integer, String, func
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="pending")
    created_at = Column(String, server_default=func.datetime('now'))
