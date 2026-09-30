"""长期记忆管理路由：走 SqliteStore，不依赖 db（独立于主问答链路）

提供记忆列表 / 手动新增 / 删除，供前端记忆管理页使用。
key 派生：手动新增对 content 取 sha1 前 12 位作稳定 key（幂等去重，
与提取链的语义 key 不冲突）。
"""
import hashlib

from fastapi import APIRouter

from app.schemas.common import ApiResponse
from app.schemas.memory import MemoryItem, MemoryCreate
from app.services.retrieval.memory_store import (
    get_memory_store, read_memories, put_memory, delete_memory,
)

router = APIRouter(prefix="/api", tags=["memory"])


@router.get("/memories", response_model=ApiResponse)
def list_memories():
    items = [MemoryItem(**m).model_dump() for m in read_memories(get_memory_store())]
    return ApiResponse(data=items)


@router.post("/memories", response_model=ApiResponse)
def create_memory(req: MemoryCreate):
    key = "m_" + hashlib.sha1(req.content.encode("utf-8")).hexdigest()[:12]
    put_memory(get_memory_store(), key, req.content, req.category)
    return ApiResponse(code="SUCCESS", message="已添加", data={"key": key})


@router.delete("/memories/{key}", response_model=ApiResponse)
def remove_memory(key: str):
    delete_memory(get_memory_store(), key)   # 不存在也返回成功
    return ApiResponse(code="SUCCESS", message="已删除")
