from sqlalchemy import Column, Integer, String, Text, func
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    file_size = Column(Integer)
    chunk_count = Column(Integer, default=0)
    status = Column(String, default="pending")  # pending/processing/ready/error/cancelled
    task_id = Column(String)  # 当前关联的异步任务 ID（checkpoint thread_id）
    visibility = Column(String, default="all")  # all=全员可见 / manager_only=经理专属
    metadata_json = Column(Text)  # 存储元数据JSON
    structure_json = Column(Text)  # 存储结构信息JSON
    created_at = Column(String, server_default=func.datetime('now'))
