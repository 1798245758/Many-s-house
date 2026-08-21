from unittest.mock import MagicMock, patch

from app.services.retrieval.agent_chain import rag_chain_query
from app.services.retrieval.intent_chain import ClassifyResult, SlotResult


def _mock_model(action: str, **slot_kwargs):
    """mock：with_structured_output 依次返回 classify/slot 两个链；
    model 本身供 answer_chain/direct_chain 管道末端当函数调用"""
    cls_mock = MagicMock()
    cls_mock.return_value = ClassifyResult(action=action, risk_reason="测试风险")

    slot_mock = MagicMock()
    slot_mock.return_value = SlotResult(**slot_kwargs)

    model = MagicMock()
    # 意图链：prompt | structured_output 管道把 mock 当函数调用
    model.with_structured_output.side_effect = [cls_mock, slot_mock]
    # 回答链：prompt | model | parser 管道同样把 model 当函数调用
    model.return_value = "直答内容"
    return model


def test_refuse_returns_refusal():
    model = _mock_model("refuse")
    resp = rag_chain_query(None, "有害问题", "sk-test", model=model)
    assert resp.response_type == "refusal"
    assert resp.intent is None


def test_clarify_returns_clarification():
    model = _mock_model("clarify", task="制度咨询", entities=["休假"],
                        clarification_question="您想了解哪个部门？")
    resp = rag_chain_query(None, "那个制度", "sk-test", model=model)
    assert resp.response_type == "clarification"
    assert resp.clarification_question == "您想了解哪个部门？"
    assert resp.intent.task == "制度咨询"


def test_direct_returns_answer_without_sources():
    model = _mock_model("direct", task="闲聊")
    resp = rag_chain_query(None, "你好", "sk-test", model=model)
    assert resp.response_type == "answer"
    assert resp.answer == "直答内容"
    assert resp.sources == []
    assert resp.intent.task == "闲聊"


def test_search_no_hits_returns_empty_notice(db_session):
    model = _mock_model("search", task="制度咨询", search_keywords=["员工", "休假"])
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=[]):
        resp = rag_chain_query(db_session, "胖东来员工休假制度", "sk-test", model=model)
    assert resp.response_type == "answer"
    assert resp.answer == "当前知识库中没有相关信息，请先上传文档。"
    assert resp.intent.task == "制度咨询"
