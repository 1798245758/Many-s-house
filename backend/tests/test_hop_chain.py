from unittest.mock import MagicMock

from app.services.retrieval.hop_chain import HopDecision, build_hop_reasoner

QUESTION = "张三公司的总部在哪个国家？"
EVIDENCE = "[来源: 员工手册]\n张三是远航科技有限公司的创始人。"
REMAINING = ["远航科技有限公司的总部在哪个国家？"]


def _mock_model(decision):
    """管道把 structured_output 链 mock 当函数调用，用 return_value"""
    chain_mock = MagicMock()
    chain_mock.return_value = decision
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock
    return model


def test_reasoner_returns_finished_decision():
    """证据充分：LLM 判定 finished，无需下一跳"""
    model = _mock_model(HopDecision(finished=True, next_query="", next_keywords=[]))
    reasoner = build_hop_reasoner(api_key="sk-test", model=model)
    out = reasoner(QUESTION, EVIDENCE, REMAINING)
    assert out.finished is True


def test_reasoner_returns_next_hop_from_evidence():
    """证据拿到公司名：推理出依赖该公司名的下一跳查询"""
    decision = HopDecision(finished=False, next_query="远航科技有限公司的总部在哪个国家？",
                           next_keywords=["远航科技", "总部", "国家"])
    model = _mock_model(decision)
    reasoner = build_hop_reasoner(api_key="sk-test", model=model)
    out = reasoner(QUESTION, EVIDENCE, REMAINING)
    assert out.finished is False
    assert out.next_keywords == ["远航科技", "总部", "国家"]
    # 推理入参应包含主问题与证据（链 mock 收到渲染后的 prompt）
    prompt = str(model.with_structured_output.return_value.call_args[0][0])
    assert QUESTION in prompt and "远航科技有限公司" in prompt


def test_reasoner_failure_degrades_to_queue_head():
    """LLM 异常且队列非空：降级为顺序消费队首子问题"""
    chain_mock = MagicMock()
    chain_mock.side_effect = RuntimeError("LLM 挂了")
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock

    reasoner = build_hop_reasoner(api_key="sk-test", model=model)
    out = reasoner(QUESTION, EVIDENCE, REMAINING)
    assert out.finished is False
    assert out.next_query == REMAINING[0]


def test_reasoner_failure_with_empty_queue_finishes():
    """LLM 异常且队列已空：直接终止循环"""
    chain_mock = MagicMock()
    chain_mock.side_effect = RuntimeError("LLM 挂了")
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock

    reasoner = build_hop_reasoner(api_key="sk-test", model=model)
    out = reasoner(QUESTION, EVIDENCE, [])
    assert out.finished is True
