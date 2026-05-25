from pydantic import BaseModel

class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int | None = None
    chunk_count: int = 0
    status: str
    created_at: str | None = None
    model_config = {"from_attributes": True}
