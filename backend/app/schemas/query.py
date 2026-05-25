from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)

class SourceInfo(BaseModel):
    chunk_id: int
    content_snippet: str
    document_name: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
