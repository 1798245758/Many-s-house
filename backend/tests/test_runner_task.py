"""任务运行器测试：提交即落盘 queued、工人续跑至终态、取消排队任务"""
import time
import pytest
from unittest.mock import MagicMock

from app.services.tasks import cancel as cancel_mod
from app.services.tasks import runner
from app.services.tasks.machine import QUEUED, TERMINAL_STATES, SUCCEEDED, CANCELLED

M = "app.services.tasks.ingest_graph"


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    """隔离的运行器：临时 checkpoint 库 + 打桩的阶段函数"""
    monkeypatch.setattr(runner, "DB_PATH", tmp_path / "task.db")
    runner.reset_runtime()
    cancel_mod.clear_all()
    monkeypatch.setattr(f"{M}.extract_document",
                        lambda path: {"content": "内容", "metadata": {}, "structure": {}})
    monkeypatch.setattr(f"{M}.clean_text", lambda text: text)
    monkeypatch.setattr(f"{M}.semantic_chunk", lambda text: ["块"])
    monkeypatch.setattr(f"{M}.vectorize_chunks", lambda chunks: [b"e"])
    monkeypatch.setattr(f"{M}.save_chunks", MagicMock())
    monkeypatch.setattr(f"{M}._save_doc_metadata", MagicMock())
    monkeypatch.setattr(f"{M}._cleanup_doc", MagicMock())
    yield
    runner.reset_runtime()
    cancel_mod.clear_all()


def _wait_terminal(task_id, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = runner.get_task_snapshot(task_id)
        if snap and snap.get("status") in TERMINAL_STATES:
            return snap
        time.sleep(0.05)
    return runner.get_task_snapshot(task_id)


def test_submit_persists_queued_then_runs_to_success(runtime):
    task_id = runner.submit_task("doc_ingest", {
        "doc_id": 1, "filename": "a.txt", "file_path": "/tmp/a.txt", "file_type": "txt",
    })
    final = _wait_terminal(task_id)
    assert final is not None
    assert final["status"] == SUCCEEDED
    # checkpoint 历史可追溯，含创建态
    history = runner.get_task_history(task_id)
    assert QUEUED in [h.get("status") for h in history]


def test_cancel_queued_task(runtime):
    """排队任务取消：checkpoint 立即置 cancelled，工人接手后走收口清理"""
    monkey_task_id = runner.submit_task("doc_ingest", {
        "doc_id": 2, "filename": "b.txt", "file_path": "/tmp/b.txt", "file_type": "txt",
    })
    # 先打满线程池无法模拟，直接取消后等待终态即可（无论排队或运行中均应到 cancelled）
    assert runner.cancel_task(monkey_task_id) == "SUCCESS"
    final = _wait_terminal(monkey_task_id)
    assert final["status"] == CANCELLED


def test_cancel_unknown_task(runtime):
    assert runner.cancel_task("nonexistent") == "NOT_FOUND"


def test_snapshot_unknown_task(runtime):
    assert runner.get_task_snapshot("nonexistent") is None
