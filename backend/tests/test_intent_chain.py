from unittest.mock import MagicMock

from app.services.retrieval.intent_chain import build_intent_gate


def _mock_model(classify_ret, slot_ret=None):
    """让 with_structured_output 按调用顺序返回 classify/slot 两个链 mock

    注意：实现中链为 prompt | structured_output 管道，invoke 时管道会
    把 mock 当函数调用，所以用 return_value 而非 invoke.return_value。
    """
    model = MagicMock()
    cls_mock = MagicMock()
    cls_mock.return_value = classify_ret
    slot_mock = MagicMock()
    slot_mock.return_value = slot_ret
    model.with_structured_output.side_effect = [cls_mock, slot_mock]
    return model


def test_refuse_action():
    from app.services.retrieval.intent_chain import ClassifyResult
    model = _mock_model(ClassifyResult(action="refuse", risk_reason="涉及违法"))
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("违法问题")
    assert out["action"] == "refuse"
    assert out["risk_reason"] == "涉及违法"
    assert out["slots"] is None


def test_search_action_extracts_keywords():
    from app.services.retrieval.intent_chain import ClassifyResult, SlotResult
    slots = SlotResult(task="制度咨询", entities=["休假"], time="", risk_note="",
                       search_keywords=["员工", "休假", "制度"], clarification_question="")
    model = _mock_model(ClassifyResult(action="search", risk_reason=""), slots)
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("胖东来的员工休假制度")
    assert out["action"] == "search"
    assert out["slots"].search_keywords == ["员工", "休假", "制度"]


def test_classify_failure_degrades_to_search():
    model = MagicMock()
    model.with_structured_output.return_value.side_effect = RuntimeError("LLM 挂了")
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("任意问题")
    assert out["action"] == "search"
    assert out["slots"].search_keywords == ["任意问题"]


def test_slot_failure_falls_back_to_question():
    from app.services.retrieval.intent_chain import ClassifyResult
    cls_mock = MagicMock()
    cls_mock.return_value = ClassifyResult(action="search", risk_reason="")
    slot_mock = MagicMock()
    slot_mock.side_effect = RuntimeError("解析失败")
    model = MagicMock()
    # with_structured_output(schema) 依次返回 classify/slot 两个 runnable
    model.with_structured_output.side_effect = [cls_mock, slot_mock]
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("任意问题")
    assert out["action"] == "search"
    assert out["slots"].search_keywords == ["任意问题"]
