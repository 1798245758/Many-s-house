"""任务取消标志：进程内集合，节点在阶段边界检查"""
import threading

_lock = threading.Lock()
_requested: set = set()


def request_cancel(task_id: str):
    with _lock:
        _requested.add(task_id)


def is_cancel_requested(task_id: str) -> bool:
    with _lock:
        return task_id in _requested


def clear_cancel(task_id: str):
    with _lock:
        _requested.discard(task_id)


def clear_all():
    with _lock:
        _requested.clear()
