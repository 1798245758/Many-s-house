from pydantic import BaseModel

class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int | None = None
    chunk_count: int = 0
    status: str
    visibility: str = "all"  # all=全员可见 / manager_only=经理专属
    created_at: str | None = None
    model_config = {"from_attributes": True}
