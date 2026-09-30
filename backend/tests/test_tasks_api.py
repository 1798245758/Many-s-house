"""任务 API 测试：上传秒回、任务查询权限、取消、SSE 事件流"""
import json
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models.document import Document


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


def test_upload_returns_task_ids_immediately(client):
    c, Session = client
    with patch("app.routers.documents.submit_task", return_value="task-abc") as st:
        res = c.post("/api/documents/upload",
                     files=[("files", ("手册.txt", b"hello", "text/plain"))],
                     headers={"X-Role": "manager"})
    body = res.json()
    assert body["code"] == "SUCCESS"
    tasks = body["data"]["tasks"]
    assert len(tasks) == 1
    assert tasks[0]["task_id"] == "task-abc"
    assert tasks[0]["filename"] == "手册.txt"
    # 文档行已建且为处理中，并记录 task_id
    db = Session()
    doc = db.query(Document).filter(Document.id == tasks[0]["document_id"]).first()
    assert doc.status == "processing"
    assert doc.task_id == "task-abc"
    db.close()
    # 提交参数携带任务类型与文档载荷
    assert st.call_args[0][0] == "doc_ingest"
    assert st.call_args[0][1]["doc_id"] == doc.id


def test_upload_rejects_bad_extension_without_task(client):
    c, _ = client
    with patch("app.routers.documents.submit_task") as st:
        res = c.post("/api/documents/upload",
                     files=[("files", ("a.exe", b"x", "application/octet-stream"))],
                     headers={"X-Role": "manager"})
    body = res.json()
    assert body["code"] == "PARTIAL_SUCCESS"
    assert body["data"]["tasks"] == []
    assert len(body["data"]["errors"]) == 1
    st.assert_not_called()


def test_get_task_requires_manager(client):
    c, _ = client
    with patch("app.services.tasks.runner.get_task_snapshot", return_value={"status": "running"}):
        body = c.get("/api/tasks/t1", headers={"X-Role": "employee"}).json()
    assert body["code"] == "PERMISSION_DENIED"


def test_get_task_returns_snapshot(client):
    c, _ = client
    snap = {"task_id": "t1", "status": "running", "stage": "embed", "progress": 60,
            "message": "向量化中", "error": "", "doc_id": 1, "filename": "a.txt",
            "task_type": "doc_ingest", "chunk_count": 0}
    with patch("app.services.tasks.runner.get_task_snapshot", return_value=snap):
        body = c.get("/api/tasks/t1", headers={"X-Role": "manager"}).json()
    assert body["code"] == "SUCCESS"
    assert body["data"]["status"] == "running"
    assert body["data"]["progress"] == 60


def test_get_task_not_found(client):
    c, _ = client
    with patch("app.services.tasks.runner.get_task_snapshot", return_value=None):
        body = c.get("/api/tasks/nope", headers={"X-Role": "manager"}).json()
    assert body["code"] == "NOT_FOUND"


def test_cancel_task(client):
    c, _ = client
    with patch("app.services.tasks.runner.cancel_task", return_value="SUCCESS") as ct:
        body = c.post("/api/tasks/t1/cancel", headers={"X-Role": "manager"}).json()
    assert body["code"] == "SUCCESS"
    ct.assert_called_once_with("t1")


def test_cancel_task_not_found(client):
    c, _ = client
    with patch("app.services.tasks.runner.cancel_task", return_value="NOT_FOUND"):
        body = c.post("/api/tasks/nope/cancel", headers={"X-Role": "manager"}).json()
    assert body["code"] == "NOT_FOUND"


def test_sse_stream_emits_progress_then_done(client, monkeypatch):
    c, _ = client
    monkeypatch.setattr("app.routers.tasks.POLL_INTERVAL", 0.01)
    running = {"task_id": "t1", "task_type": "doc_ingest", "status": "running", "stage": "chunk",
               "progress": 45, "message": "切块中", "error": "", "doc_id": 1,
               "filename": "a.txt", "chunk_count": 0}
    done = {**running, "status": "succeeded", "stage": "index", "progress": 100, "message": "完成"}
    with patch("app.services.tasks.runner.get_task_snapshot", side_effect=[running, done]):
        with c.stream("GET", "/api/tasks/t1/stream?role=manager") as res:
            text = "".join(res.iter_text())
    assert "event: progress" in text
    assert "event: done" in text
    done_line = [l for l in text.splitlines() if l.startswith("data")][-1]
    payload = json.loads(done_line[len("data: "):])
    assert payload["status"] == "succeeded"


def test_sse_stream_rejects_employee(client):
    c, _ = client
    res = c.get("/api/tasks/t1/stream?role=employee")
    assert res.status_code == 403
