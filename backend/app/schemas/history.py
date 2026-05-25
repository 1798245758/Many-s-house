from pydantic import BaseModel

class HistoryOut(BaseModel):
    id: int
    query_text: str
    answer_text: str
    sources: str | None = None
    created_at: str | None = None
    model_config = {"from_attributes": True}
