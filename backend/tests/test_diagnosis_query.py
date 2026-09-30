"""检索/问答错误诊断测试：证据采集、分流分析、多轮上下文泛化、API 权限与列表"""
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.query_trace import QueryTrace
from app.services.diagnosis import agent


def _db():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return sessionmaker(bind=eng)()


# ---------- Task 7: 采集问答证据 + 分流分析 ----------

def test_collect_query_evidence():
    db = _db()
    db.add(QueryTrace(id=1, conversation_id="c1", question="问", answer_snippet="答",
                      response_type="answer", final_level="error", final_code="E1",
                      steps=json.dumps([{"step": "rerank", "level": "error", "code": "E1",
                                         "message": "候选集为空", "metrics": {"ranked": 0}}])))
    db.commit()
    ev = agent.collect_query_evidence(db, 1)
    assert ev["subject_type"] == "query" and ev["final_code"] == "E1"
    assert ev["steps"][0]["step"] == "rerank"
    assert agent.collect_query_evidence(db, 999) is None


def test_analyze_query_uses_retrieval_prompt():
    db = _db()
    db.add(QueryTrace(id=2, conversation_id="c", question="q", response_type="answer",
                      final_level="error", final_code="E1", steps="[]"))
    db.commit()
    ev = agent.collect_query_evidence(db, 2)

    class FakeClient:
        def __init__(self):
            self.sys = ""

        def ask(self, system, user):
            self.sys = system
            return "根因报告"

    c = FakeClient()
    out = agent.analyze(ev, c)
    assert out["analysis"] == "根因报告" and out["trace_id"] == 2
    assert "检索" in c.sys


def test_analyze_task_side_keeps_final_status():
    """分流后任务侧 analyze 不回归：仍返回 task_id / final_status"""
    ev = {"task_id": "t1", "task_type": "doc_ingest", "final_status": "failed",
          "final_stage": "extract", "error": "文件损坏", "retry_count": 0,
          "events": [], "document": None, "error_log": ""}

    class FakeClient:
        def ask(self, system, user):
            return "任务根因"

    out = agent.analyze(ev, FakeClient())
    assert out["task_id"] == "t1" and out["final_status"] == "failed"
    assert out["analysis"] == "任务根因" and out["llm_error"] == ""


# ---------- Task 8: 多轮上下文主体泛化 ----------

def test_ask_query_subject_pins_evidence():
    from app.services.diagnosis import context as ctx
    ctx.clear_sessions()
    ev = {"subject_type": "query", "trace_id": 7, "question": "q", "final_level": "error",
          "final_code": "E1", "steps": [{"step": "rerank", "level": "error", "code": "E1",
                                          "message": "候选集为空"}], "response_type": "answer",
          "answer_snippet": "", "conversation_id": "c"}

    class FakeClient:
        def messages(self, msgs, temperature=0.3):
            return "ok"

        def ask(self, system, user, temperature=0.3):
            return "摘要"

    out = ctx.ask(7, "为什么", ev, FakeClient(), session_id=None, subject_type="query")
    assert out["session_id"] and out["task_id"] == 7 and out["answer"] == "ok"
    mgr = ctx._sessions.get(out["session_id"])
    assert mgr.subject_type == "query" and "候选集为空" in mgr.pinned_evidence


def test_task_subject_still_works_positional():
    """既有任务侧位置调用 ContextManager(task_id, evidence, client) 不回归"""
    from app.services.diagnosis import context as ctx
    ev = {"task_id": "t1", "task_type": "doc_ingest", "final_status": "failed",
          "final_stage": "extract", "error": "文件损坏", "retry_count": 0,
          "events": [], "document": None, "error_log": ""}

    class FakeClient:
        def messages(self, msgs, temperature=0.3):
            return "任务答"

        def ask(self, system, user, temperature=0.3):
            return "摘要"

    mgr = ctx.ContextManager("t1", ev, FakeClient())
    assert mgr.task_id == "t1" and mgr.subject_id == "t1" and mgr.subject_type == "task"
    assert "文件损坏" in mgr.pinned_evidence


# ---------- Task 9: 诊断路由与 schema ----------

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import get_db
from app.models.setting import Setting


@pytest.fixture
def client_api():
    eng = create_engine("sqlite:///:memory:",
                        connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=eng)
    Session = sessionmaker(bind=eng)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), Session
    app.dependency_overrides.clear()


def _seed_key(Session):
    db = Session()
    db.add(Setting(key="api_key", value="sk-test"))
    db.commit()
    db.close()


def test_query_errors_manager_only(client_api):
    c, _ = client_api
    r = c.get("/api/diagnosis/query-errors", headers={"X-Role": "employee"})
    assert r.json()["code"] == "PERMISSION_DENIED"


def test_query_errors_lists_rows(client_api):
    c, Session = client_api
    db = Session()
    db.add(QueryTrace(id=1, conversation_id="c", question="q", response_type="answer",
                      final_level="error", final_code="E1", steps="[]"))
    db.commit()
    db.close()
    data = c.get("/api/diagnosis/query-errors", headers={"X-Role": "manager"}).json()["data"]
    assert any(x["trace_id"] == 1 and x["final_level"] == "error" for x in data)


def test_query_errors_level_filter(client_api):
    c, Session = client_api
    db = Session()
    db.add(QueryTrace(id=1, conversation_id="c", question="q1", response_type="answer",
                      final_level="error", final_code="E1", steps="[]"))
    db.add(QueryTrace(id=2, conversation_id="c", question="q2", response_type="answer",
                      final_level="normal", final_code="N1", steps="[]"))
    db.commit()
    db.close()
    data = c.get("/api/diagnosis/query-errors?level=error",
                 headers={"X-Role": "manager"}).json()["data"]
    assert [x["trace_id"] for x in data] == [1]


def test_analyze_query_api_happy_path(client_api):
    c, Session = client_api
    _seed_key(Session)
    db = Session()
    db.add(QueryTrace(id=5, conversation_id="c", question="q", response_type="answer",
                      final_level="error", final_code="E1",
                      steps=json.dumps([{"step": "rerank", "level": "error",
                                         "code": "E1", "message": "候选集为空"}])))
    db.commit()
    db.close()
    with patch("app.services.deepseek.DeepSeekClient.ask", return_value="## 根因分析\n无命中"):
        body = c.post("/api/diagnosis/analyze-query", json={"trace_id": 5},
                      headers={"X-Role": "manager"}).json()
    assert body["code"] == "SUCCESS"
    assert "根因分析" in body["data"]["analysis"]
    assert body["data"]["evidence"]["final_code"] == "E1"


def test_analyze_query_api_not_found(client_api):
    c, Session = client_api
    _seed_key(Session)
    body = c.post("/api/diagnosis/analyze-query", json={"trace_id": 999},
                  headers={"X-Role": "manager"}).json()
    assert body["code"] == "NOT_FOUND"


def test_chat_query_flow(client_api):
    c, Session = client_api
    from app.services.diagnosis import context as ctx
    ctx.clear_sessions()
    _seed_key(Session)
    db = Session()
    db.add(QueryTrace(id=8, conversation_id="c", question="q", response_type="answer",
                      final_level="error", final_code="E1", steps="[]"))
    db.commit()
    db.close()
    with patch("app.services.deepseek.DeepSeekClient.messages", return_value="检索根因"):
        r1 = c.post("/api/diagnosis/chat",
                    json={"question": "为什么", "subject_type": "query", "trace_id": 8},
                    headers={"X-Role": "manager"}).json()
    assert r1["code"] == "SUCCESS" and r1["data"]["answer"] == "检索根因"
    sid = r1["data"]["session_id"]
    with patch("app.services.deepseek.DeepSeekClient.messages", return_value="修复建议"):
        r2 = c.post("/api/diagnosis/chat",
                    json={"question": "怎么修", "session_id": sid,
                          "subject_type": "query", "trace_id": 8},
                    headers={"X-Role": "manager"}).json()
    assert r2["data"]["session_id"] == sid and r2["data"]["context_stats"]["total_turns"] == 2
