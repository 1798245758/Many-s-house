from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    conversation_id: str | None = None       # 会话标识（短期记忆的 thread_id，前端生成）

class SourceInfo(BaseModel):
    chunk_id: int
    content_snippet: str
    document_name: str
    page: int | None = None                  # 来源页码（PDF 逐页切块时有值，支撑 citation 页级溯源）

class IntentInfo(BaseModel):
    """意图分析结果（任务/实体/时间/风险）"""
    task: str
    entities: list[str] = []
    time: str | None = None
    risk_note: str | None = None

class QueryResponse(BaseModel):
    response_type: str = "answer"            # answer | clarification | refusal | permission_denied
    answer: str                              # 回答 / 澄清引导语 / 拒绝说明 / 权限不足说明
    sources: list[SourceInfo] = []
    intent: IntentInfo | None = None
    clarification_question: str | None = None
    evidence_status: str | None = None       # verified | uncertain（仅 search 分支，证据校验结果）
