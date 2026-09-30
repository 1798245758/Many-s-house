"""入库任务状态图测试：流转、失败、取消、向量化阻塞重试、checkpoint 历史"""
import pytest
from unittest.mock import MagicMock, patch
from langgraph.checkpoint.memory import MemorySaver

from app.services.tasks import cancel as cancel_mod
from app.services.tasks.ingest_graph import build_ingest_graph
from app.services.tasks.machine import SUCCEEDED, FAILED, CANCELLED, BLOCKED, QUEUED, RUNNING
from app.services.tasks.state import make_initial_state

M = "app.services.tasks.ingest_graph"


@pytest.fixture(autouse=True)
def _clear_cancel():
    cancel_mod.clear_all()
    yield
    cancel_mod.clear_all()


@pytest.fixture
def graph():
    return build_ingest_graph(MemorySaver())


def _initial(task_id="t1"):
    return make_initial_state(task_id, "doc_ingest", {
        "doc_id": 1, "filename": "x.docx", "file_path": "/tmp/x.docx", "file_type": "docx",
    })


def _config(task_id="t1"):
    return {"configurable": {"thread_id": task_id}}


def _patch_stages(monkeypatch, vectorize=None, extract=None):
    """打桩各阶段工作函数，隔离真实解析/向量化/写库"""
    monkeypatch.setattr(f"{M}.extract_document",
                        extract or (lambda path: {"content": "正文内容", "metadata": {"a": 1}, "structure": {"h": []}}))
    monkeypatch.setattr(f"{M}.clean_text", lambda text: text.strip())
    monkeypatch.setattr(f"{M}.semantic_chunk", lambda text: ["块一", "块二"])
    monkeypatch.setattr(f"{M}.vectorize_chunks", vectorize or (lambda chunks: [b"e1", b"e2"]))
    monkeypatch.setattr(f"{M}.save_chunks", MagicMock())
    monkeypatch.setattr(f"{M}._save_doc_metadata", MagicMock())
    monkeypatch.setattr(f"{M}._cleanup_doc", MagicMock())


def test_full_pipeline_success(graph, monkeypatch):
    _patch_stages(monkeypatch)
    final = graph.invoke(_initial(), _config())
    assert final["status"] == SUCCEEDED
    assert final["progress"] == 100
    assert final["chunk_count"] == 2
    assert final["stage"] == "index"


def test_checkpoint_history_records_transitions(graph, monkeypatch):
    """checkpoint 历史即状态流转日志：应包含 queued→running→succeeded"""
    _patch_stages(monkeypatch)
    graph.invoke(_initial(), _config())
    statuses = [s.values.get("status") for s in graph.get_state_history(_config())]
    assert QUEUED in statuses
    assert RUNNING in statuses
    assert SUCCEEDED in statuses


def test_extract_failure_marks_failed(graph, monkeypatch):
    def boom(path):
        raise ValueError("文件损坏")
    _patch_stages(monkeypatch, extract=boom)
    final = graph.invoke(_initial(), _config())
    assert final["status"] == FAILED
    assert "文件损坏" in final["error"]


def test_cancel_before_start(graph, monkeypatch):
    _patch_stages(monkeypatch)
    cancel_mod.request_cancel("t1")
    final = graph.invoke(_initial(), _config())
    assert final["status"] == CANCELLED


def test_embed_blocked_then_recovery(graph, monkeypatch):
    """向量化首次异常→blocked，重试成功→最终 succeeded"""
    monkeypatch.setattr(f"{M}.EMBED_RETRY_DELAY", 0)
    vec = MagicMock(side_effect=[RuntimeError("限流"), [b"e1", b"e2"]])
    _patch_stages(monkeypatch, vectorize=vec)
    final = graph.invoke(_initial(), _config())
    assert final["status"] == SUCCEEDED
    assert vec.call_count == 2
    statuses = [s.values.get("status") for s in graph.get_state_history(_config())]
    assert BLOCKED in statuses


def test_embed_retries_exhausted_marks_failed(graph, monkeypatch):
    monkeypatch.setattr(f"{M}.EMBED_RETRY_DELAY", 0)
    vec = MagicMock(side_effect=RuntimeError("持续限流"))
    _patch_stages(monkeypatch, vectorize=vec)
    final = graph.invoke(_initial(), _config())
    assert final["status"] == FAILED
    assert "持续限流" in final["error"]
