from sqlalchemy import Column, Integer, String, Text, func
from app.database import Base


class QueryTrace(Base):
    """RAG 检索/问答每轮的多步轨迹与错误分级（面向诊断页，独立于 QueryHistory）"""
    __tablename__ = "query_trace"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, index=True)      # 会话（thread_id），未传兜底 "default"
    question = Column(Text, nullable=False)
    answer_snippet = Column(Text)                      # 截断 _ANSWER_SNIPPET_LEN
    response_type = Column(String)                     # answer|clarification|refusal|permission_denied|error(E5整轮失败)
    final_level = Column(String, index=True)           # normal|degraded|error
    final_code = Column(String)                        # 触发 final_level 的最高级错误码
    steps = Column(Text)                               # JSON：多步轨迹数组
    created_at = Column(String, server_default=func.datetime('now'))
