from pydantic import BaseModel
from typing import Optional, Any

class ApiResponse(BaseModel):
    code: str = "SUCCESS"
    message: str = "操作成功"
    data: Optional[Any] = None
