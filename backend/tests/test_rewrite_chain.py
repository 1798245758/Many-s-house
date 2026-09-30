from unittest.mock import MagicMock

from app.services.retrieval.rewrite_chain import RewriteResult, build_query_rewriter


def test_rewrite_returns_structured_result():
    """正常路径：一次结构化调用产出改写问题/子问题/关键词"""
    ret = RewriteResult(
        rewritten_question="胖东来员工的休假制度和培训流程分别是什么？",
        sub_questions=["胖东来员工的休假制度是什么？", "胖东来员工的培训流程是什么？"],
        keywords=["胖东来", "员工", "休假制度", "培训流程"],
    )
    chain_mock = MagicMock()
    chain_mock.return_value = ret  # 管道把 mock 当函数调用，用 return_value
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock

    rewriter = build_query_rewriter(api_key="sk-test", model=model)
    out = rewriter("那个休假和培训是咋弄的")
    assert out.rewritten_question == "胖东来员工的休假制度和培训流程分别是什么？"
    assert len(out.sub_questions) == 2
    assert out.keywords == ["胖东来", "员工", "休假制度", "培训流程"]


def test_rewrite_failure_degrades_to_original_question():
    """LLM 异常：降级为原问题 + 空子问题 + 原问题作唯一关键词"""
    chain_mock = MagicMock()
    chain_mock.side_effect = RuntimeError("LLM 挂了")
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock

    rewriter = build_query_rewriter(api_key="sk-test", model=model)
    out = rewriter("任意问题")
    assert out.rewritten_question == "任意问题"
    assert out.sub_questions == []
    assert out.keywords == ["任意问题"]


def test_rewrite_prompt_contains_history_and_status():
    """带短期记忆：渲染后 prompt 含近期对话与上轮任务状态（指代消解/澄清承接）"""
    chain_mock = MagicMock()
    chain_mock.return_value = RewriteResult(
        rewritten_question="胖东来的工资标准是什么？", sub_questions=[], keywords=["工资"])
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock

    rewriter = build_query_rewriter(api_key="sk-test", model=model)
    rewriter("那工资呢？",
             history=["问：胖东来的休假制度是什么？\n答：年假为…"],
             task_status="已完成回答")
    prompt = str(chain_mock.call_args[0][0])
    assert "胖东来的休假制度是什么？" in prompt
    assert "已完成回答" in prompt
    assert "那工资呢？" in prompt


def test_rewrite_prompt_without_history_uses_placeholder():
    """无历史：占位渲染，调用方式不变"""
    chain_mock = MagicMock()
    chain_mock.return_value = RewriteResult(
        rewritten_question="问题", sub_questions=[], keywords=["问题"])
    model = MagicMock()
    model.with_structured_output.return_value = chain_mock

    rewriter = build_query_rewriter(api_key="sk-test", model=model)
    rewriter("问题")
    prompt = str(chain_mock.call_args[0][0])
    assert "（无）" in prompt
