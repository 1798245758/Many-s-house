"""任务状态定义：全纯数据（str/int/list/bytes），ORM 对象绝不进图，规避 checkpoint 序列化坑"""
from typing import TypedDict

from app.services.tasks.machine import QUEUED


class TaskState(TypedDict, total=False):
    # 生命周期
    task_id: str
    task_type: str
    status: str       # machine 中的六态之一
    stage: str        # 当前阶段：extract/clean/chunk/embed/index
    progress: int     # 0-100
    message: str      # 面向用户的进度描述
    error: str        # 失败原因
    # 业务载荷（文档入库）
    doc_id: int
    filename: str
    file_path: str
    file_type: str
    # 阶段间纯数据中间产物
    raw_text: str
    cleaned_text: str
    chunks: list
    embeddings: list
    chunk_count: int
    retry_count: int


def make_initial_state(task_id: str, task_type: str, payload: dict) -> dict:
    """创建任务的初始状态（排队态）"""
    return {
        "task_id": task_id,
        "task_type": task_type,
        "status": QUEUED,
        "stage": "",
        "progress": 0,
        "message": "任务已创建，等待执行",
        "error": "",
        "retry_count": 0,
        **payload,
    }
