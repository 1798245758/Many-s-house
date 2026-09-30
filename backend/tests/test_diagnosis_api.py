"""错误诊断 Agent 测试：checkpoint 证据采集、LLM 分析、诊断 API 权限与降级"""
import time
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models.document import Document
from app.models.setting import Setting
from app.services.diagnosis import agent
from app.services.tasks import cancel as cancel_mod
from app.services.tasks import runner
from app.services.tasks.machine import FAILED, TERMINAL_STATES

M = "app.services.tasks.ingest_graph"


# ---------- 真实 checkpoint 采集（隔离运行器 + 临时 SqliteSaver） ----------

@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "DB_PATH", tmp_path / "task.db")
    runner.reset_runtime()
    cancel_mod.clear_all()
    monkeypatch.setattr(f"{M}.extract_document", lambda path: (_ for _ in ()).throw(
        RuntimeError("PDF 解析失败：文件损坏")))
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


def test_collect_evidence_from_real_checkpoint(runtime, db_session):
    """失败任务的状态机流转历史即诊断证据"""
    doc = Document(filename="坏文件.pdf", file_type="pdf", file_size=10, status="error")
    db_session.add(doc)
    db_session.commit()
    task_id = runner.submit_task("doc_ingest", {
        "doc_id": doc.id, "filename": "坏文件.pdf", "file_path": "/tmp/x.pdf", "file_type": "pdf",
    })
    final = _wait_terminal(task_id)
    assert final["status"] == FAILED

    ev = agent.collect_evidence(db_session, task_id, error_log="traceback ...")
    assert ev["final_status"] == FAILED
    assert ev["final_stage"] == "extract"
    assert "PDF 解析失败" in ev["error"]
    assert ev["document"]["filename"] == "坏文件.pdf"
    assert ev["error_log"] == "traceback ..."
    statuses = [e["status"] for e in ev["events"]]
    assert "queued" in statuses and FAILED in statuses
    # 大体积中间产物不进证据
    assert all("raw_text" not in e and "chunks" not in e for e in ev["events"])


def test_collect_evidence_unknown_task(runtime, db_session):
    assert agent.collect_evidence(db_session, "nonexistent") is None


def test_analyze_calls_llm_and_degrades(runtime, db_session):
    task_id = runner.submit_task("doc_ingest", {
        "doc_id": 1, "filename": "a.pdf", "file_path": "/tmp/a.pdf", "file_type": "pdf"})
    _wait_terminal(task_id)
    ev = agent.collect_evidence(db_session, task_id)

    ok_client = MagicMock()
    ok_client.ask.return_value = "## 根因分析\n文件损坏"
    result = agent.analyze(ev, ok_client)
    assert result["analysis"] == "## 根因分析\n文件损坏"
    assert result["llm_error"] == ""
    # 证据文本包含事件日志与错误信息
    prompt = ok_client.ask.call_args[0][1]
    assert "PDF 解析失败" in prompt and "状态机流转事件日志" in prompt

    bad_client = MagicMock()
    bad_client.ask.side_effect = RuntimeError("timeout")
    degraded = agent.analyze(ev, bad_client)
    assert degraded["analysis"] == ""
    assert "timeout" in degraded["llm_error"]
    assert degraded["final_status"] == FAILED


# ---------- 诊断 API ----------

@pytest.fixture
def client():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), Session
    app.dependency_overrides.clear()


def test_diagnosis_apis_require_manager(client):
    c, _ = client
    assert c.get("/api/diagnosis/tasks", headers={"X-Role": "employee"}).json()["code"] == "PERMISSION_DENIED"
    assert c.get("/api/diagnosis/tasks/t1/evidence", headers={"X-Role": "employee"}).json()["code"] == "PERMISSION_DENIED"
    body = c.post("/api/diagnosis/analyze", json={"task_id": "t1"},
                  headers={"X-Role": "employee"}).json()
    assert body["code"] == "PERMISSION_DENIED"


def test_list_diagnosable_tasks(client):
    c, Session = client
    db = Session()
    db.add(Document(filename="坏.pdf", file_type="pdf", status="error", task_id="t-bad"))
    db.add(Document(filename="好.pdf", file_type="pdf", status="ready", task_id="t-ok"))
    db.add(Document(filename="旧.pdf", file_type="pdf", status="error", task_id=None))
    db.commit()
    db.close()
    body = c.get("/api/diagnosis/tasks", headers={"X-Role": "manager"}).json()
    assert body["code"] == "SUCCESS"
    task_ids = [t["task_id"] for t in body["data"]]
    assert task_ids == ["t-bad"]  # 仅失败/取消且有 checkpoint 的任务


def test_evidence_api_not_found(client):
    c, _ = client
    with patch("app.services.tasks.runner.get_task_snapshot", return_value=None):
        body = c.get("/api/diagnosis/tasks/nope/evidence", headers={"X-Role": "manager"}).json()
    assert body["code"] == "NOT_FOUND"


def test_evidence_api_returns_snapshot(client):
    c, _ = client
    ev = {"task_id": "t1", "final_status": "failed", "events": []}
    with patch("app.services.tasks.runner.get_task_snapshot", return_value={"status": "failed", "doc_id": None}), \
         patch("app.services.tasks.runner.get_task_history", return_value=[]):
        body = c.get("/api/diagnosis/tasks/t1/evidence", headers={"X-Role": "manager"}).json()
    assert body["code"] == "SUCCESS"
    assert body["data"]["task_id"] == ev["task_id"]
    assert body["data"]["final_status"] == "failed"


def test_analyze_api_happy_path(client):
    c, Session = client
    db = Session()
    db.add(Setting(key="api_key", value="sk-test"))
    db.commit()
    db.close()
    with patch("app.services.tasks.runner.get_task_snapshot",
               return_value={"status": "failed", "stage": "embed", "error": "向量服务超时", "doc_id": None}), \
         patch("app.services.tasks.runner.get_task_history", return_value=[]), \
         patch("app.services.deepseek.DeepSeekClient.ask", return_value="## 根因分析\n向量服务不可用"):
        body = c.post("/api/diagnosis/analyze", json={"task_id": "t1"},
                      headers={"X-Role": "manager"}).json()
    assert body["code"] == "SUCCESS"
    assert "根因分析" in body["data"]["analysis"]
    assert body["data"]["evidence"]["final_status"] == "failed"


def test_analyze_api_missing_key_and_llm_error(client):
    c, Session = client
    snap_patch = patch("app.services.tasks.runner.get_task_snapshot",
                       return_value={"status": "failed", "doc_id": None})
    hist_patch = patch("app.services.tasks.runner.get_task_history", return_value=[])
    with snap_patch, hist_patch:
        body = c.post("/api/diagnosis/analyze", json={"task_id": "t1"},
                      headers={"X-Role": "manager"}).json()
    assert body["code"] == "API_KEY_MISSING"

    db = Session()
    db.add(Setting(key="api_key", value="sk-test"))
    db.commit()
    db.close()
    with patch("app.services.tasks.runner.get_task_snapshot",
               return_value={"status": "failed", "doc_id": None}), \
         patch("app.services.tasks.runner.get_task_history", return_value=[]), \
         patch("app.services.deepseek.DeepSeekClient.ask", side_effect=RuntimeError("boom")):
        body = c.post("/api/diagnosis/analyze", json={"task_id": "t1"},
                      headers={"X-Role": "manager"}).json()
    assert body["code"] == "LLM_ERROR"
    assert "boom" in body["message"]
