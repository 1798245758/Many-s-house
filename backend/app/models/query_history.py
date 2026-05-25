from sqlalchemy import Column, Integer, String, Text, func
from app.database import Base


class QueryHistory(Base):
    __tablename__ = "query_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    query_text = Column(Text, nullable=False)
    answer_text = Column(Text, nullable=False)
    sources = Column(Text)
    created_at = Column(String, server_default=func.datetime('now'))
