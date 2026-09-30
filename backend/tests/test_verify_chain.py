from unittest.mock import MagicMock

from app.services.retrieval.verify_chain import VerifyResult, build_evidence_verifier

QUESTION = "胖东来的温暖基金标准是什么？"
SUB_QUESTIONS = ["温暖基金的发放条件是什么？"]
EVIDENCE = "[来源: 胖东来温暖基金标准.md]\n员工直系亲属重病可申请温暖基金，标准为一万元。"


def _mock_model(result):
    """管道把 structured_output 链 mock 当函数调用，用 return_value"""
    chain_mock = MagicMock()
    chain_mock.return_value = result
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock
    return model


def test_verify_passes_when_four_conditions_met():
    """四条件满足：sufficient=True"""
    model = _mock_model(VerifyResult(sufficient=True))
    verifier = build_evidence_verifier(api_key="sk-test", model=model)
    out = verifier(QUESTION, SUB_QUESTIONS, EVIDENCE)
    assert out.sufficient is True


def test_verify_fails_with_missing_and_alternative_query():
    """条件不满足：给出缺失方面 + 换角度的补充查询，且入参含证据与子问题"""
    result = VerifyResult(sufficient=False, missing_aspects=["温暖基金的发放条件"],
                          next_query="温暖基金申请流程和发放规定",
                          next_keywords=["温暖基金", "发放", "申请条件"])
    model = _mock_model(result)
    verifier = build_evidence_verifier(api_key="sk-test", model=model)
    out = verifier(QUESTION, SUB_QUESTIONS, EVIDENCE)
    assert out.sufficient is False
    assert out.missing_aspects == ["温暖基金的发放条件"]
    assert out.next_keywords == ["温暖基金", "发放", "申请条件"]
    prompt = str(model.with_structured_output.return_value.call_args[0][0])
    assert QUESTION in prompt and "发放条件" in prompt and "一万元" in prompt


def test_verify_failure_degrades_to_pass():
    """LLM 异常：降级放行（视为充分），避免校验故障卡死循环"""
    chain_mock = MagicMock()
    chain_mock.side_effect = RuntimeError("LLM 挂了")
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock

    verifier = build_evidence_verifier(api_key="sk-test", model=model)
    out = verifier(QUESTION, SUB_QUESTIONS, EVIDENCE)
    assert out.sufficient is True
