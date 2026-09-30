"""任务运行器：线程池执行 + SqliteSaver 持久化 + 快照/取消/重启兜底

设计约定：
- 每个任务一个 thread_id，状态机流转全部经 Checkpoint 落盘；
- 提交时先写 queued checkpoint（创建即留痕），工人线程从断点续跑；
- 任务类型 → 图工厂 的注册表，文档入库（doc_ingest）为首个类型。
"""
import sqlite3
import uuid
from concurrent.futures import ThreadPoolExecutor

from langgraph.checkpoint.sqlite import SqliteSaver

from app.config import DB_PATH
from app.services.tasks import cancel as cancel_mod
from app.services.tasks.ingest_graph import build_ingest_graph
from app.services.tasks.machine import CANCELLED, FAILED, QUEUED, TERMINAL_STATES
from app.services.tasks.state import make_initial_state

_executor = ThreadPoolExecutor(max_workers=2)
_saver = None
_graphs = {}
_registry = {}  # task_id -> task_type


def reset_runtime():
    """重置运行器单例（测试用）"""
    global _saver, _graphs, _registry
    _saver = None
    _graphs = {}
    _registry = {}


def get_checkpointer(db_path=None) -> SqliteSaver:
    """SqliteSaver 进程级单例：多线程共享连接，autocommit 交由 saver 管理"""
    global _saver
    if _saver is None:
        path = str(db_path or DB_PATH)
        conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        _saver = SqliteSaver(conn)
        _saver.setup()
    return _saver


def _get_graph(task_type: str):
    if task_type not in _graphs:
        if task_type != "doc_ingest":
            raise ValueError(f"未知任务类型: {task_type}")
        _graphs[task_type] = build_ingest_graph(get_checkpointer())
    return _graphs[task_type]


def _config(task_id: str) -> dict:
    return {"configurable": {"thread_id": task_id}}


def submit_task(task_type: str, payload: dict) -> str:
    """创建任务：先落盘 queued checkpoint，再提交线程池执行"""
    task_id = uuid.uuid4().hex
    graph = _get_graph(task_type)
    _registry[task_id] = task_type
    initial = make_initial_state(task_id, task_type, payload)
    graph.update_state(_config(task_id), initial, as_node="__start__")
    _executor.submit(_run_task, task_id, task_type)
    return task_id


def _run_task(task_id: str, task_type: str):
    graph = _get_graph(task_type)
    config = _config(task_id)
    try:
        # 从 queued checkpoint 断点续跑
        graph.invoke(None, config)
    except Exception as e:
        # 兜底：未预期异常也保证终态落盘
        try:
            graph.update_state(config, {"status": FAILED, "error": str(e), "message": "任务执行异常"})
        except Exception:
            pass


def get_task_snapshot(task_id: str) -> dict | None:
    """读取任务最新 checkpoint 状态；不存在返回 None"""
    task_type = _registry.get(task_id, "doc_ingest")
    try:
        graph = _get_graph(task_type)
    except ValueError:
        return None
    snap = graph.get_state(_config(task_id))
    values = dict(snap.values) if snap and snap.values else None
    return values or None


def get_task_history(task_id: str) -> list[dict]:
    """checkpoint 历史（状态流转事件日志），从早到晚"""
    task_type = _registry.get(task_id, "doc_ingest")
    graph = _get_graph(task_type)
    snapshots = list(graph.get_state_history(_config(task_id)))
    return [dict(s.values) for s in reversed(snapshots) if s.values]


def cancel_task(task_id: str) -> str:
    """请求取消：排队任务立即改 checkpoint；运行中任务在下一阶段边界生效"""
    snap = get_task_snapshot(task_id)
    if not snap:
        return "NOT_FOUND"
    if snap.get("status") in TERMINAL_STATES:
        return "ALREADY_TERMINAL"
    cancel_mod.request_cancel(task_id)
    if snap.get("status") == QUEUED:
        graph = _get_graph(_registry.get(task_id, "doc_ingest"))
        graph.update_state(_config(task_id), {"status": CANCELLED, "message": "任务已取消"})
    return "SUCCESS"


def cleanup_stale_processing():
    """进程重启兜底：把残留处理中的文档回置失败（v1 不做断点续跑）"""
    from app.database import get_session_local
    from app.models.document import Document
    db = get_session_local()()
    try:
        count = db.query(Document).filter(Document.status == "processing").update({"status": "error"})
        db.commit()
        return count
    finally:
        db.close()
