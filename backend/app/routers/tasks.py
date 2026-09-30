"""异步任务 API：快照查询、SSE 进度推流、取消

SSE 鉴权说明：浏览器 EventSource 无法自定义请求头，故角色走 ?role= 查询参数，
缺省/非法按 employee（最小权限）。
"""
import json
import time

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.schemas.common import ApiResponse
from app.services.role import ROLE_MANAGER, get_role, parse_role
from app.services.tasks import runner
from app.services.tasks.machine import TERMINAL_STATES

router = APIRouter(prefix="/api/tasks", tags=["tasks"])
POLL_INTERVAL = 0.5  # SSE 轮询 checkpoint 的间隔（秒）

_PUBLIC_FIELDS = ("task_id", "task_type", "status", "stage", "progress",
                  "message", "error", "doc_id", "filename", "chunk_count")


def _public(values: dict) -> dict:
    return {k: values.get(k) for k in _PUBLIC_FIELDS}


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.get("/{task_id}", response_model=ApiResponse)
def get_task(task_id: str, role: str = Depends(get_role)):
    """读取任务最新状态（刷新页面/断线兜底）"""
    if role != ROLE_MANAGER:
        return ApiResponse(code="PERMISSION_DENIED", message="仅经理可查看任务进度")
    snap = runner.get_task_snapshot(task_id)
    if not snap:
        return ApiResponse(code="NOT_FOUND", message="任务不存在")
    return ApiResponse(data=_public(snap))


@router.get("/{task_id}/stream")
def stream_task(task_id: str, role: str = Query("employee")):
    """SSE 推流：状态变化即推 progress 事件，终态推 done 后关流"""
    if parse_role(role) != ROLE_MANAGER:
        raise HTTPException(status_code=403, detail="仅经理可查看任务进度")

    def event_stream():
        last = None
        while True:
            snap = runner.get_task_snapshot(task_id)
            if not snap:
                yield _sse("done", {"task_id": task_id, "status": "failed", "error": "任务不存在"})
                return
            payload = _public(snap)
            if payload != last:
                done = payload.get("status") in TERMINAL_STATES
                yield _sse("done" if done else "progress", payload)
                last = payload
                if done:
                    return
            time.sleep(POLL_INTERVAL)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/{task_id}/cancel", response_model=ApiResponse)
def cancel_task(task_id: str, role: str = Depends(get_role)):
    """请求取消任务：排队任务立即取消，运行中任务在下一阶段边界生效"""
    if role != ROLE_MANAGER:
        return ApiResponse(code="PERMISSION_DENIED", message="仅经理可取消任务")
    result = runner.cancel_task(task_id)
    if result == "NOT_FOUND":
        return ApiResponse(code="NOT_FOUND", message="任务不存在")
    if result == "ALREADY_TERMINAL":
        return ApiResponse(code="PARAM_ERROR", message="任务已结束，无需取消")
    return ApiResponse(message="取消请求已提交")
