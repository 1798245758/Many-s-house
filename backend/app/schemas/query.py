from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

class SourceInfo(BaseModel):
    chunk_id: int
    content_snippet: str
    document_name: str

class IntentInfo(BaseModel):
    """意图分析结果（任务/实体/时间/风险）"""
    task: str
    entities: list[str] = []
    time: str | None = None
    risk_note: str | None = None

class QueryResponse(BaseModel):
    response_type: str = "answer"            # answer | clarification | refusal
    answer: str                              # 回答 / 澄清引导语 / 拒绝说明
    sources: list[SourceInfo] = []
    intent: IntentInfo | None = None
    clarification_question: str | None = None
