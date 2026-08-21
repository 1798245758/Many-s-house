from app.schemas.query import QueryResponse, IntentInfo


def test_intent_info_defaults():
    info = IntentInfo(task="制度咨询")
    assert info.entities == []
    assert info.time is None
    assert info.risk_note is None


def test_query_response_backward_compatible():
    r = QueryResponse(answer="ok")
    assert r.response_type == "answer"
    assert r.sources == []
    assert r.intent is None
    assert r.clarification_question is None


def test_clarification_response_fields():
    intent = IntentInfo(task="制度咨询", entities=["休假"], time="2024年")
    r = QueryResponse(
        response_type="clarification",
        answer="请补充信息",
        intent=intent,
        clarification_question="您想了解哪个部门的休假制度？",
    )
    d = r.model_dump()
    assert d["response_type"] == "clarification"
    assert d["intent"]["task"] == "制度咨询"
    assert d["intent"]["entities"] == ["休假"]
    assert d["clarification_question"] == "您想了解哪个部门的休假制度？"
