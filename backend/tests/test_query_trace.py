"""检索轨迹埋点（多步 trace）、三级分级、收口落库 QueryTrace 的测试"""
from unittest.mock import patch

from app.models.query_trace import QueryTrace  # 顶部导入确保 db_session 建表时已注册
from app.schemas.query import QueryResponse
from app.services.retrieval.agent_chain import (
    _step, _classify_trace, _record_trace, build_rag_graph, rag_chain_query)
from app.services.retrieval.hop_chain import HopDecision
from app.services.retrieval.verify_chain import VerifyResult
from tests.test_agent_chain import _mock_model, _Chunk


# ===== _step / _classify_trace =====

def test_step_builds_pure_record():
    rec = _step("rerank", "error", code="E1", message="候选集为空", ranked=0)
    assert rec["step"] == "rerank" and rec["level"] == "error"
    assert rec["code"] == "E1" and rec["metrics"] == {"ranked": 0} and rec["ts"]


def test_classify_trace_takes_max_level():
    trace = [_step("intent", "normal"), _step("verify", "degraded", code="W1"),
             _step("rerank", "error", code="E1")]
    assert _classify_trace(trace) == ("error", "E1")


def test_classify_trace_same_level_takes_last():
    trace = [_step("a", "degraded", code="W2"), _step("b", "degraded", code="W5")]
    assert _classify_trace(trace) == ("degraded", "W5")


def test_classify_trace_empty_is_normal():
    assert _classify_trace([]) == ("normal", "")


# ===== 节点埋点（跑真实图，检查 final["trace"]）=====

def _run(model, question, db, role="employee", thread="t-trace"):
    graph = build_rag_graph("sk-test", model=model)
    return graph.invoke({"question": question, "role": role},
                        config={"configurable": {"db": db, "thread_id": thread}})


def _codes(final):
    return [s["code"] for s in final["trace"]]


def test_empty_candidates_marks_E1(db_session):
    model = _mock_model("search", task="制度咨询", search_keywords=["休假"])
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=[]):
        final = _run(model, "胖东来休假制度", db_session)
    assert "E1" in _codes(final)
    assert _classify_trace(final["trace"])[0] == "error"


def test_permission_denied_marks_N2_normal():
    model = _mock_model("search", task="培训", search_keywords=["运营经理"])
    final = _run(model, "运营经理是怎么培训的？", None, role="employee")
    assert "N2" in _codes(final)
    assert _classify_trace(final["trace"])[0] == "normal"


def test_refuse_marks_N3():
    model = _mock_model("refuse")
    assert "N3" in _codes(_run(model, "有害问题", None))


def test_direct_marks_N5():
    model = _mock_model("direct", task="闲聊")
    assert "N5" in _codes(_run(model, "你好", None))


def test_verified_answer_marks_N1(db_session):
    model = _mock_model("search", hop_decisions=[HopDecision(finished=True)],
                        verify_results=[VerifyResult(sufficient=True)],
                        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=[_Chunk(1)]):
        final = _run(model, "温暖基金咋申请", db_session)
    assert "N1" in _codes(final)
    assert _classify_trace(final["trace"])[0] == "normal"


def test_verify_uncertain_marks_W1(db_session):
    insufficient = VerifyResult(sufficient=False, missing_aspects=["发放标准"],
                                next_query="换角度", next_keywords=["换角度"])
    model = _mock_model("search", hop_decisions=[HopDecision(finished=True)],
                        verify_results=[insufficient, insufficient],
                        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=[_Chunk(1)]):
        final = _run(model, "温暖基金咋申请", db_session)
    assert "W1" in _codes(final)
    assert _classify_trace(final["trace"])[0] == "degraded"


def test_rewrite_degraded_marks_W2(db_session):
    model = _mock_model("search", task="制度咨询", search_keywords=["休假"])
    model.rw_mock.side_effect = RuntimeError("改写挂了")
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=[]):
        final = _run(model, "休假制度", db_session)
    assert "W2" in _codes(final)


def test_trace_resets_between_turns(db_session):
    """同一 thread 两轮：入口重置，第二轮 trace 不累积第一轮"""
    model = _mock_model("direct", task="闲聊")
    graph = build_rag_graph("sk-test", model=model)
    cfg = {"configurable": {"db": db_session, "thread_id": "reset-conv"}}
    f1 = graph.invoke({"question": "你好", "role": "employee"}, config=cfg)
    f2 = graph.invoke({"question": "再见", "role": "employee"}, config=cfg)
    assert len(f2["trace"]) == len(f1["trace"])


# ===== 收口落库 _record_trace =====

def test_record_trace_writes_row(db_session):
    resp = QueryResponse(response_type="answer", answer="答", evidence_status="uncertain")
    _record_trace(db_session, "c1", "问", resp, [_step("rerank", "error", code="E1")])
    row = db_session.query(QueryTrace).first()
    assert row.final_level == "error" and row.final_code == "E1"
    assert row.conversation_id == "c1" and row.response_type == "answer"


def test_record_trace_db_none_no_raise():
    resp = QueryResponse(response_type="answer", answer="a")
    _record_trace(None, "c", "q", resp, [])  # db 为 None 时静默跳过，不抛异常


# ===== query.py 整轮失败落 E5 =====

def test_query_router_records_E5_on_total_failure(monkeypatch, db_session):
    from app.models.setting import Setting
    import app.routers.query as qr
    from app.schemas.query import QueryRequest

    db_session.add(Setting(key="api_key", value="sk-test"))
    db_session.commit()

    def _boom(*a, **k):
        raise RuntimeError("boom")

    monkeypatch.setattr(qr, "generate_answer_via_chain", _boom)
    monkeypatch.setattr(qr, "generate_answer", _boom)
    resp = qr.ask_query(QueryRequest(question="问"), db=db_session, role="employee")
    assert resp.code == "LLM_ERROR"
    row = db_session.query(QueryTrace).first()
    assert row and row.final_code == "E5" and row.final_level == "error"


def test_query_router_records_E5_on_chain_fail_legacy_ok(monkeypatch, db_session):
    """图失败但旧流水线兜底成功：仍落 E5（链路级故障可诊断），响应 SUCCESS"""
    from app.models.setting import Setting
    import app.routers.query as qr
    from app.schemas.query import QueryRequest

    db_session.add(Setting(key="api_key", value="sk-test"))
    db_session.commit()

    def _boom(*a, **k):
        raise RuntimeError("graph down")

    monkeypatch.setattr(qr, "generate_answer_via_chain", _boom)
    monkeypatch.setattr(qr, "generate_answer",
                        lambda *a, **k: QueryResponse(response_type="answer", answer="兜底答"))
    resp = qr.ask_query(QueryRequest(question="问"), db=db_session, role="employee")
    assert resp.code == "SUCCESS"
    row = db_session.query(QueryTrace).first()
    assert row and row.final_code == "E5" and row.final_level == "error"


# ===== 收口 answer 分支落库（回归 Blocker：answer 分支曾漏 _record_trace）=====

def test_answer_branch_records_normal_trace(db_session):
    model = _mock_model("search", hop_decisions=[HopDecision(finished=True)],
                        verify_results=[VerifyResult(sufficient=True)],
                        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=[_Chunk(1)]):
        resp = rag_chain_query(db_session, "温暖基金咆申请", "sk-test", model=model,
                               conversation_id="c-normal")
    assert resp.response_type == "answer"
    row = db_session.query(QueryTrace).filter(QueryTrace.conversation_id == "c-normal").first()
    assert row and row.final_level == "normal" and row.final_code == "N1"


def test_answer_branch_records_degraded_trace(db_session):
    """uncertain 硬停作答走 answer 分支：W1 降级必须落库可见"""
    insufficient = VerifyResult(sufficient=False, missing_aspects=["发放标准"],
                                next_query="换角度", next_keywords=["换角度"])
    model = _mock_model("search", hop_decisions=[HopDecision(finished=True)],
                        verify_results=[insufficient, insufficient],
                        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=[_Chunk(1)]):
        rag_chain_query(db_session, "温暖基金咆申请", "sk-test", model=model,
                        conversation_id="c-degraded")
    row = db_session.query(QueryTrace).filter(QueryTrace.conversation_id == "c-degraded").first()
    assert row and row.final_level == "degraded" and row.final_code == "W1"
