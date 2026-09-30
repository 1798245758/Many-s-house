"""诊断请求 Schema"""
from typing import Optional

from pydantic import BaseModel


class DiagnosisRequest(BaseModel):
    task_id: str
    error_log: Optional[str] = ""  # 用户可选粘贴的附加错误日志


class DiagnosisChatRequest(BaseModel):
    """多轮诊断追问：首轮可不带 session_id（服务端创建并回传）"""
    task_id: str = ""
    question: str
    session_id: Optional[str] = None
    error_log: Optional[str] = ""  # 仅首轮建会话时用于采集证据
    subject_type: str = "task"     # task=入库任务 | query=检索问答
    trace_id: Optional[int] = None  # subject_type=query 时的轨迹 ID


class QueryAnalyzeRequest(BaseModel):
    """检索错误分析：按轨迹 ID 采集证据并交 LLM 分析根因"""
    trace_id: int
    error_log: Optional[str] = ""
