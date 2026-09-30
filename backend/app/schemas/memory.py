from typing import Literal

from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """一条长期记忆（确定性事实）"""
    key: str
    content: str
    category: str = "preference"          # role | instruction | format | preference
    updated_at: str = ""


class MemoryCreate(BaseModel):
    """手动新增记忆入参（key 由后端按 content 派生，幂等去重）"""
    content: str = Field(..., min_length=1)
    category: Literal["role", "instruction", "format", "preference"] = "preference"
