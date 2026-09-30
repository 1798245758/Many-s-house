"""各检索链在 LLM 失败走兜底时应回传 degraded=True（供检索轨迹打 W 级降级码）"""
from unittest.mock import MagicMock

from app.services.retrieval.rewrite_chain import build_query_rewriter
from app.services.retrieval.hop_chain import build_hop_reasoner
from app.services.retrieval.verify_chain import build_evidence_verifier
from app.services.retrieval.intent_chain import build_intent_gate


def _failing_model():
    # 管道把结构化输出 mock 当函数直接调用，故用 side_effect 触发异常（与既有链测试一致）
    chain_mock = MagicMock()
    chain_mock.side_effect = RuntimeError("llm down")
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock
    return model


def test_rewrite_degraded_on_llm_error():
    out = build_query_rewriter("k", model=_failing_model())("原问题", history=[], task_status="none")
    assert out.degraded is True


def test_hop_degraded_on_llm_error():
    out = build_hop_reasoner("k", model=_failing_model())("q", "evidence", ["子问题"])
    assert out.degraded is True


def test_verify_degraded_on_llm_error():
    out = build_evidence_verifier("k", model=_failing_model())("q", ["子问题"], "evidence")
    assert out.degraded is True and out.sufficient is True


def test_intent_degraded_on_llm_error():
    out = build_intent_gate("k", model=_failing_model())("问")
    assert out["degraded"] is True and out["action"] == "search"
