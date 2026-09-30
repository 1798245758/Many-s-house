from pydantic import BaseModel

class DocumentOut(BaseModel):
    id: int
    filename: str
    file_type: str
    file_size: int | None = None
    chunk_count: int = 0
    status: str
    visibility: str = "all"  # all=全员可见 / manager_only=经理专属
    task_id: str | None = None  # 处理中关联的异步任务，前端据此重连进度流
    created_at: str | None = None
    model_config = {"from_attributes": True}
