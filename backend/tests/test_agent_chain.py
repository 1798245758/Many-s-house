from unittest.mock import MagicMock, patch

from app.services.retrieval.agent_chain import rag_chain_query
from app.services.retrieval.intent_chain import ClassifyResult, SlotResult
from app.services.retrieval.rewrite_chain import RewriteResult
from app.services.retrieval.hop_chain import HopDecision
from app.services.retrieval.verify_chain import VerifyResult
from app.services.retrieval.memory_chain import MemoryUpdateResult
from app.config import HOP_MAX_HOPS, HOP_TOP_K_PER_HOP

DEFAULT_REWRITE = RewriteResult(rewritten_question="改写后的规范问题", sub_questions=[],
                                keywords=["改写", "关键词"])


class _Chunk:
    """最小候选对象：仅需 id/content/document/page/document_id 属性"""
    def __init__(self, chunk_id, content="内容", page=None, document_id=None):
        self.id = chunk_id
        self.content = content
        self.document = None
        self.page = page
        self.document_id = document_id


def _mock_model(action: str, rewrite=None, hop_decisions=None,
                verify_results=None, **slot_kwargs):
    """mock：with_structured_output 按 schema 动态路由（与真实"每类 schema 构建
    一条链"的语义一致），天然支持多轮调用；model 本身供 answer_chain/
    direct_chain 管道末端当函数调用。
    hop_decisions：多跳推理链返回值序列（可含异常对象）。
    verify_results：证据校验链返回值序列（可含异常对象），缺省单次通过。"""
    rw_mock = MagicMock()
    rw_mock.return_value = rewrite or DEFAULT_REWRITE

    cls_mock = MagicMock()
    cls_mock.return_value = ClassifyResult(action=action, risk_reason="测试风险")

    slot_mock = MagicMock()
    slot_mock.return_value = SlotResult(**slot_kwargs)

    # 长期记忆提取链：本文件测试不关注提取，固定返回空操作（避免误路由消耗 verify mock）
    mem_mock = MagicMock()
    mem_mock.return_value = MemoryUpdateResult()

    model = MagicMock()
    # hop/verify 链调用轮数不定（多跳/补检），存列表用指针取；
    # pop 会清空列表，测试末尾无法再按序断言入参，故用指针
    model.hop_mocks, model.verify_mocks = [], []
    for d in (hop_decisions or []):
        hop_mock = MagicMock()
        if isinstance(d, Exception):
            hop_mock.side_effect = d
        else:
            hop_mock.return_value = d
        model.hop_mocks.append(hop_mock)
    for v in (verify_results or [VerifyResult(sufficient=True)]):
        verify_mock = MagicMock()
        if isinstance(v, Exception):
            verify_mock.side_effect = v
        else:
            verify_mock.return_value = v
        model.verify_mocks.append(verify_mock)
    model.hop_idx = [0]
    model.verify_idx = [0]
    schema_map = {RewriteResult: rw_mock, ClassifyResult: cls_mock,
                  SlotResult: slot_mock, MemoryUpdateResult: mem_mock}

    def _route_structured(schema, method=None):
        if schema in schema_map:
            return schema_map[schema]
        if schema is HopDecision:
            mock = model.hop_mocks[model.hop_idx[0]]
            model.hop_idx[0] += 1
            return mock
        mock = model.verify_mocks[model.verify_idx[0]]
        model.verify_idx[0] += 1
        return mock

    model.with_structured_output.side_effect = _route_structured
    # 保留链引用供测试断言入参（side_effect 被调用后变成迭代器不可下标）
    model.rw_mock, model.cls_mock, model.slot_mock = rw_mock, cls_mock, slot_mock
    # 回答链：prompt | model | parser 管道同样把 model 当函数调用
    model.return_value = "直答内容"
    return model


def test_refuse_returns_refusal():
    model = _mock_model("refuse")
    resp = rag_chain_query(None, "有害问题", "sk-test", model=model)
    assert resp.response_type == "refusal"
    assert resp.intent is None
    # 意图先行：refuse 短路直接结束，改写链不触发（省一次 LLM 调用）
    model.rw_mock.assert_not_called()


def test_clarify_returns_clarification():
    model = _mock_model("clarify", task="制度咨询", entities=["休假"],
                        clarification_question="您想了解哪个部门？")
    resp = rag_chain_query(None, "那个制度", "sk-test", model=model)
    assert resp.response_type == "clarification"
    assert resp.clarification_question == "您想了解哪个部门？"
    assert resp.intent.task == "制度咨询"
    # 意图先行：clarify 短路直接结束，改写链不触发（省一次 LLM 调用）
    model.rw_mock.assert_not_called()


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
    # 无命中时不应触发 LLM 回答调用（model 本体未被当函数调用）
    model.assert_not_called()


def test_search_sources_carry_page_number(db_session):
    """citation 页级溯源：命中 chunk 带页码时 sources 回传 page"""
    model = _mock_model("search", hop_decisions=[HopDecision(finished=True)],
                        verify_results=[VerifyResult(sufficient=True)],
                        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               return_value=[_Chunk(1, "第7页的内容", page=7)]):
        resp = rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    assert resp.sources[0].page == 7


def test_employee_manager_topic_returns_permission_denied():
    """员工提问命中经理关键词：确定性门禁短路，不调任何 LLM"""
    model = _mock_model("search", task="培训咨询", search_keywords=["运营经理", "培训"])
    resp = rag_chain_query(None, "运营经理是怎么培训的？", "sk-test", model=model)
    assert resp.response_type == "permission_denied"
    assert "权限不足" in resp.answer
    # 门禁在意图关卡之前：意图/改写链均未触发（长期记忆提取会调 with_structured_output，
    # 故按具体链 mock 断言，而非全局 with_structured_output）
    model.cls_mock.assert_not_called()
    model.slot_mock.assert_not_called()
    model.rw_mock.assert_not_called()
    model.assert_not_called()


def test_manager_same_question_not_blocked():
    """经理问同样的问题不被门禁拦截，正常进入意图链"""
    model = _mock_model("direct", task="培训咨询")
    resp = rag_chain_query(None, "运营经理是怎么培训的？", "sk-test", model=model, role="manager")
    assert resp.response_type == "answer"
    assert resp.answer == "直答内容"


def test_search_query_merges_rewrite_and_slot_keywords(db_session):
    """search 分支：检索串为改写关键词 + 槽位关键词合并去重"""
    rewrite = RewriteResult(rewritten_question="胖东来员工休假制度是什么？",
                            sub_questions=[], keywords=["胖东来", "休假"])
    model = _mock_model("search", rewrite=rewrite, task="制度咨询",
                        search_keywords=["休假", "制度"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               return_value=[]) as mock_hs:
        rag_chain_query(db_session, "休假是咋规定的", "sk-test", model=model)
    # hybrid_search(db, embedding, query, ...)：第 3 个位置参数为检索串，
    # 去重后应为 改写关键词 + 槽位新增关键词（重复的"休假"不重复出现）
    assert mock_hs.call_args[0][2] == "胖东来 休假 制度"
    # 意图先行且解耦：意图分类基于原始问题而非改写后问题（cls_mock 收到渲染后的 prompt）
    cls_prompt = str(model.cls_mock.call_args[0][0])
    assert "休假是咋规定的" in cls_prompt
    assert "胖东来员工休假制度是什么？" not in cls_prompt


def test_direct_uses_rewritten_question():
    """direct 分支：直答链使用改写后的问题作答"""
    rewrite = RewriteResult(rewritten_question="你好，请介绍一下你自己",
                            sub_questions=[], keywords=["问候"])
    model = _mock_model("direct", rewrite=rewrite, task="闲聊")
    rag_chain_query(None, "你谁啊", "sk-test", model=model)
    # model 本体被管道当函数调用，入参渲染后的 prompt 应含改写后问题
    assert "你好，请介绍一下你自己" in str(model.call_args)


def test_multihop_accumulates_hits_and_finishes(db_session):
    """多跳：推理链 next→finish，两跳命中按 chunk_id 去重累积并统一重排"""
    rewrite = RewriteResult(
        rewritten_question="张三公司的总部在哪个国家？",
        sub_questions=["张三的公司是哪家？", "该公司的总部在哪个国家？"],
        keywords=["张三", "公司"])
    model = _mock_model(
        "search", rewrite=rewrite,
        hop_decisions=[
            HopDecision(finished=False, next_query="远航科技总部",
                        next_keywords=["远航科技", "总部"]),
            HopDecision(finished=True),
        ],
        task="多跳咨询", search_keywords=["总部"])
    hop1, hop2 = [_Chunk(1, "张三是远航科技的创始人")], [_Chunk(2, "远航科技总部在中国"), _Chunk(1)]
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               side_effect=[hop1, hop2]) as mock_hs, \
         patch("app.services.retrieval.agent_chain.vector_store.get_embeddings",
               return_value={1: [1.0, 0.0], 2: [0.0, 1.0]}):
        resp = rag_chain_query(db_session, "张三公司总部在哪", "sk-test", model=model)
    assert resp.response_type == "answer"
    assert resp.answer == "直答内容"
    # 两跳检索：首跳合并关键词串 + 推理出的下一跳关键词串；每跳 top_k 取配置值；
    # chunk 1 重复命中被去重，最终来源 2 个（重排后截断保留）
    assert mock_hs.call_count == 2
    assert mock_hs.call_args_list[0][0][2] == "张三 公司 总部"
    assert mock_hs.call_args_list[1][0][2] == "远航科技 总部"
    assert mock_hs.call_args_list[0][1]["top_k"] == HOP_TOP_K_PER_HOP
    assert len(resp.sources) == 2
    assert resp.evidence_status == "verified"


def test_multihop_stops_at_max_hops(db_session):
    """推理链持续 next：跳数达 HOP_MAX_HOPS 强制退出循环"""
    rewrite = RewriteResult(
        rewritten_question="主问题",
        sub_questions=["子问题1", "子问题2", "子问题3", "子问题4"],
        keywords=["关键词"])
    model = _mock_model(
        "search", rewrite=rewrite,
        hop_decisions=[HopDecision(finished=False, next_query=f"跳{i}",
                                   next_keywords=[f"跳{i}"]) for i in range(3)],
        task="多跳咨询", search_keywords=[])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               return_value=[]) as mock_hs:
        resp = rag_chain_query(db_session, "主问题", "sk-test", model=model)
    assert mock_hs.call_count == HOP_MAX_HOPS
    assert resp.answer == "当前知识库中没有相关信息，请先上传文档。"


def test_hop_reason_failure_degrades_to_queue(db_session):
    """推理链异常：降级顺序消费子问题队列，队列耗尽后退出"""
    rewrite = RewriteResult(rewritten_question="主问题",
                            sub_questions=["子问题一", "子问题二"],
                            keywords=["关键词"])
    model = _mock_model("search", rewrite=rewrite,
                        hop_decisions=[RuntimeError("LLM 挂了"), RuntimeError("又挂了")],
                        task="多跳咨询", search_keywords=[])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               return_value=[]) as mock_hs:
        rag_chain_query(db_session, "主问题", "sk-test", model=model)
    # 首跳 + 两次降级跳（各消费一个子问题），第 4 个查询串应为子问题二；
    # 队列耗尽后不再进入推理（否则 side_effect 会报 StopIteration）
    assert mock_hs.call_count == 3
    assert mock_hs.call_args_list[1][0][2] == "子问题一"
    assert mock_hs.call_args_list[2][0][2] == "子问题二"


def test_verify_insufficient_retries_then_passes(db_session):
    """校验不满足：写入换角度补检查询继续循环，二次校验通过 → verified"""
    model = _mock_model(
        "search",
        hop_decisions=[HopDecision(finished=True)],
        verify_results=[
            VerifyResult(sufficient=False, missing_aspects=["发放标准"],
                         next_query="温暖基金标准", next_keywords=["温暖基金", "标准"]),
            VerifyResult(sufficient=True),
        ],
        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               side_effect=[[_Chunk(1)], [_Chunk(2)]]) as mock_hs:
        resp = rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    # 首跳 + 一次补检；补检查询串来自校验链的补充关键词（非子问题队列）
    assert mock_hs.call_count == 2
    assert mock_hs.call_args_list[1][0][2] == "温暖基金 标准"
    assert resp.evidence_status == "verified"
    assert resp.answer == "直答内容"
    # 首次校验应收到带来源标注的证据与改写后问题（渲染后的 prompt）
    vf_prompt = str(model.verify_mocks[0].call_args[0][0])
    assert "改写后的规范问题" in vf_prompt
    assert "内容" in vf_prompt


def test_verify_hard_stop_on_missing_streak(db_session):
    """连续相同缺失方面达 VERIFY_MISSING_STREAK → 硬停：
    uncertain + 对冲指令注入回答 prompt"""
    insufficient = VerifyResult(sufficient=False, missing_aspects=["发放标准"],
                                next_query="换角度查询",
                                next_keywords=["换角度", "查询"])
    model = _mock_model(
        "search",
        hop_decisions=[HopDecision(finished=True)],
        verify_results=[insufficient, insufficient],
        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               side_effect=[[_Chunk(1)], [_Chunk(2)]]) as mock_hs:
        resp = rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    # 首跳 + 一次补检后硬停（缺失方面连续相同），不再继续循环
    assert mock_hs.call_count == 2
    assert resp.evidence_status == "uncertain"
    # 回答 prompt 注入对冲指令（不确定语气 + 缺失方面）
    assert "不确定语气" in str(model.call_args)
    assert "发放标准" in str(model.call_args)


def test_multiturn_history_injected_into_rewrite(db_session):
    """同一 conversation_id 两轮：第二轮改写 prompt 含第一轮问答与任务状态"""
    model = _mock_model("direct", task="闲聊")
    conv = "conv-mt-1"
    rag_chain_query(db_session, "胖东来的休假制度是什么？", "sk-test",
                    model=model, conversation_id=conv)
    rag_chain_query(db_session, "那工资呢？", "sk-test",
                    model=model, conversation_id=conv)
    assert model.rw_mock.call_count == 2
    prompt2 = str(model.rw_mock.call_args_list[1][0][0])
    assert "胖东来的休假制度是什么？" in prompt2
    assert "直答内容" in prompt2
    assert "已完成回答" in prompt2


def test_multiturn_isolated_between_conversations(db_session):
    """不同 conversation_id：短期记忆互不串"""
    model = _mock_model("direct", task="闲聊")
    rag_chain_query(db_session, "问题甲", "sk-test", model=model,
                    conversation_id="conv-iso-A")
    rag_chain_query(db_session, "问题乙", "sk-test", model=model,
                    conversation_id="conv-iso-B")
    prompt_b = str(model.rw_mock.call_args_list[1][0][0])
    assert "问题甲" not in prompt_b


def test_multiturn_resets_loop_state(db_session):
    """第二轮 search：上轮 checkpoint 残留的循环状态必须重置，
    否则第二轮多跳被跳过"""
    rewrite = RewriteResult(rewritten_question="主问题",
                            sub_questions=["子问题"], keywords=["关键词"])
    model = _mock_model(
        "search", rewrite=rewrite,
        hop_decisions=[HopDecision(finished=True),
                       HopDecision(finished=False, next_query="跳2",
                                   next_keywords=["跳2"])],
        verify_results=[VerifyResult(sufficient=True), VerifyResult(sufficient=True)],
        task="多跳咨询", search_keywords=["关键词"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               side_effect=[[_Chunk(1)], [_Chunk(2)], [_Chunk(3)]]) as mock_hs:
        rag_chain_query(db_session, "第一轮问题", "sk-test", model=model,
                        conversation_id="conv-loop")
        rag_chain_query(db_session, "第二轮问题", "sk-test", model=model,
                        conversation_id="conv-loop")
    # 第一轮 1 次检索；第二轮若未重置 loop_finished 会首跳后直接重排（共 2 次），
    # 正常推进多跳应为 3 次
    assert mock_hs.call_count == 3


def test_verify_failure_degrades_to_verified(db_session):
    """校验链异常：降级视为充分放行，不卡死循环"""
    model = _mock_model(
        "search",
        hop_decisions=[HopDecision(finished=True)],
        verify_results=[RuntimeError("校验链挂了")],
        task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               return_value=[_Chunk(1)]) as mock_hs:
        resp = rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    assert mock_hs.call_count == 1
    assert resp.evidence_status == "verified"
    assert resp.answer == "直答内容"


def test_compiled_graph_is_singleton_per_api_key():
    """同一 api_key 复用同一编译图；Key 变更重建"""
    from app.services.retrieval.agent_chain import get_compiled_graph
    g1 = get_compiled_graph("sk-singleton-A")
    g2 = get_compiled_graph("sk-singleton-A")
    assert g1 is g2
    assert get_compiled_graph("sk-singleton-B") is not g1


def test_search_receives_db_from_runtime_config(db_session):
    """单例图不捕获请求级 Session：db 经 invoke 的 RunnableConfig 注入，
    检索节点拿到的是本次请求的 db"""
    model = _mock_model("search", task="制度咨询", search_keywords=["休假"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               return_value=[]) as mock_hs:
        rag_chain_query(db_session, "休假制度", "sk-test", model=model)
    # hybrid_search(db, embedding, query, ...)：第 1 个位置参数为注入的 db
    assert mock_hs.call_args[0][0] is db_session


def test_page_readback_on_uncertain(db_session):
    """证据不足硬停时，context 节点用 page_tools.read_page 回读 Top 命中原文页，
    回读内容进入回答 prompt 的上下文（按需回读）"""
    insufficient = VerifyResult(sufficient=False, missing_aspects=["发放标准"],
                                next_query="换角度查询", next_keywords=["换角度", "查询"])
    model = _mock_model(
        "search", hop_decisions=[HopDecision(finished=True)],
        verify_results=[insufficient, insufficient],
        task="制度咨询", search_keywords=["温暖基金"])
    hits = [_Chunk(1, "温暖基金片段", page=3, document_id=42)]
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=hits), \
         patch("app.services.retrieval.agent_chain.read_page",
               return_value={"document_id": 42, "page": 3,
                             "content": "温暖基金完整原文页内容", "chunk_ids": [1]}) as rp:
        resp = rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    assert resp.evidence_status == "uncertain"
    assert rp.called  # 证据不足触发了回读
    # 回读原文页拼入上下文，随 prompt 传给回答模型
    assert "温暖基金完整原文页内容" in str(model.call_args)
    assert "原文回读" in str(model.call_args)


def test_no_page_readback_when_verified(db_session):
    """证据充分（verified）时不回读原文页，避免上下文无谓膨胀"""
    model = _mock_model(
        "search", hop_decisions=[HopDecision(finished=True)],
        verify_results=[VerifyResult(sufficient=True)],
        task="制度咨询", search_keywords=["温暖基金"])
    hits = [_Chunk(1, "内容", page=3, document_id=42)]
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=hits), \
         patch("app.services.retrieval.agent_chain.read_page") as rp:
        resp = rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    assert resp.evidence_status == "verified"
    rp.assert_not_called()


def test_page_readback_skips_chunks_without_page(db_session):
    """命中 chunk 无页码（非 PDF 逐页切块）时不触发回读"""
    insufficient = VerifyResult(sufficient=False, missing_aspects=["发放标准"],
                                next_query="换角度", next_keywords=["换角度"])
    model = _mock_model(
        "search", hop_decisions=[HopDecision(finished=True)],
        verify_results=[insufficient, insufficient],
        task="制度咨询", search_keywords=["温暖基金"])
    hits = [_Chunk(1, "无页码内容", page=None, document_id=42)]
    with patch("app.services.retrieval.agent_chain.hybrid_search", return_value=hits), \
         patch("app.services.retrieval.agent_chain.read_page") as rp:
        resp = rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    assert resp.evidence_status == "uncertain"
    rp.assert_not_called()
