"""长期记忆测试：提取链结构化输出、apply 增删改、全量注入、跨会话生效、
手动删除、提取失败降级、上限裁剪（Store 用 conftest 注入的 InMemoryStore）"""
from unittest.mock import MagicMock, patch

from langgraph.store.memory import InMemoryStore

from app.services.retrieval import memory_store
from app.services.retrieval.memory_store import (
    get_memory_store, put_memory, read_memories, format_memories, delete_memory)
from app.services.retrieval.memory_chain import (
    build_memory_extractor, apply_memory_ops, MemoryOp, MemoryUpdateResult)
from app.services.retrieval.agent_chain import rag_chain_query
from app.services.retrieval.intent_chain import ClassifyResult, SlotResult
from app.services.retrieval.rewrite_chain import RewriteResult
from app.services.retrieval.verify_chain import VerifyResult

DEFAULT_REWRITE = RewriteResult(rewritten_question="改写后的规范问题",
                                sub_questions=[], keywords=["关键词"])


class _Chunk:
    def __init__(self, chunk_id, content="内容", page=None, document_id=None):
        self.id = chunk_id
        self.content = content
        self.document = None
        self.page = page
        self.document_id = document_id


def _mock_model(action="direct", rewrite=None, memory_result=None, **slot_kwargs):
    """mock：with_structured_output 按 schema 路由，含 MemoryUpdateResult（提取链）"""
    rw_mock = MagicMock(); rw_mock.return_value = rewrite or DEFAULT_REWRITE
    cls_mock = MagicMock(); cls_mock.return_value = ClassifyResult(action=action, risk_reason="")
    slot_mock = MagicMock(); slot_mock.return_value = SlotResult(**slot_kwargs)
    mem_mock = MagicMock(); mem_mock.return_value = memory_result or MemoryUpdateResult()
    verify_mock = MagicMock(); verify_mock.return_value = VerifyResult(sufficient=True)
    model = MagicMock()
    schema_map = {RewriteResult: rw_mock, ClassifyResult: cls_mock,
                  SlotResult: slot_mock, MemoryUpdateResult: mem_mock}
    model.with_structured_output.side_effect = \
        lambda schema, method=None: schema_map.get(schema, verify_mock)
    model.rw_mock, model.mem_mock = rw_mock, mem_mock
    model.return_value = "直答内容"
    return model


# ===== 提取链 =====

def test_extractor_parses_operations():
    """提取链结构化输出正确解析为操作列表"""
    ret = MemoryUpdateResult(operations=[
        MemoryOp(op="create", key="user_role", content="用户自称经理",
                 category="role", reason="用户明示身份")])
    chain_mock = MagicMock(); chain_mock.return_value = ret
    model = MagicMock(); model.with_structured_output.return_value = chain_mock

    extractor = build_memory_extractor("sk-test", model=model)
    out = extractor("我是经理", "好的", existing=[])
    assert len(out.operations) == 1
    assert out.operations[0].key == "user_role"
    assert out.operations[0].category == "role"


def test_extractor_failure_degrades_to_empty():
    """提取链异常：降级为空操作，不抛出"""
    chain_mock = MagicMock(); chain_mock.side_effect = RuntimeError("LLM 挂了")
    model = MagicMock(); model.with_structured_output.return_value = chain_mock

    extractor = build_memory_extractor("sk-test", model=model)
    out = extractor("问题", "回答", existing=[])
    assert out.operations == []


# ===== apply 增删改 =====

def test_apply_memory_ops_create_update_delete():
    store = get_memory_store()
    apply_memory_ops(store, [MemoryOp(op="create", key="k1", content="v1", category="role")])
    assert [m["content"] for m in read_memories(store)] == ["v1"]
    # update 覆盖同 key
    apply_memory_ops(store, [MemoryOp(op="update", key="k1", content="v2", category="role")])
    assert [m["content"] for m in read_memories(store)] == ["v2"]
    # delete
    apply_memory_ops(store, [MemoryOp(op="delete", key="k1")])
    assert read_memories(store) == []


def test_apply_ignores_invalid_and_missing():
    store = get_memory_store()
    # create 缺 content 被忽略；delete 不存在 key 静默跳过
    apply_memory_ops(store, [MemoryOp(op="create", key="k", content=None),
                             MemoryOp(op="delete", key="nope")])
    assert read_memories(store) == []


# ===== 读写辅助 =====

def test_format_memories_placeholder_and_lines():
    assert format_memories([]) == "（无）"
    text = format_memories([{"category": "format", "content": "用表格回答"}])
    assert text == "- [format] 用表格回答"


def test_trim_over_max(monkeypatch):
    """超上限按 updated_at 删最旧"""
    monkeypatch.setattr(memory_store, "MEMORY_MAX_ITEMS", 3)
    store = InMemoryStore()
    for i in range(5):
        store.put(memory_store.NAMESPACE, f"k{i}",
                  {"content": f"c{i}", "category": "preference",
                   "updated_at": f"2026-01-0{i + 1}"})
    memory_store._trim(store)
    kept = [m["key"] for m in read_memories(store)]
    assert kept == ["k2", "k3", "k4"]   # 删掉 updated_at 最旧的 k0/k1


# ===== 全量注入 =====

def test_memory_injected_into_rewrite():
    put_memory(get_memory_store(), "user_role", "用户是餐饮部员工", "role")
    model = _mock_model("direct", task="闲聊")
    rag_chain_query(None, "你好", "sk-test", model=model)
    assert "用户是餐饮部员工" in str(model.rw_mock.call_args)


def test_memory_injected_into_direct_answer():
    put_memory(get_memory_store(), "answer_style", "回答要简洁", "instruction")
    model = _mock_model("direct", task="闲聊")
    rag_chain_query(None, "你好", "sk-test", model=model)
    assert "回答要简洁" in str(model.call_args)


def test_memory_injected_into_search_answer(db_session):
    put_memory(get_memory_store(), "fmt", "用表格回答", "format")
    model = _mock_model("search", task="制度咨询", search_keywords=["温暖基金"])
    with patch("app.services.retrieval.agent_chain.hybrid_search",
               return_value=[_Chunk(1, "温暖基金内容")]):
        rag_chain_query(db_session, "温暖基金咋申请", "sk-test", model=model)
    assert "用表格回答" in str(model.call_args)


def test_no_memory_uses_placeholder():
    model = _mock_model("direct", task="闲聊")
    rag_chain_query(None, "你好", "sk-test", model=model)
    assert "（无）" in str(model.rw_mock.call_args)


# ===== 跨会话 / 删除 =====

def test_memory_persists_across_sessions(db_session):
    """A 会话提取写入的记忆，B 会话（新 conversation_id）仍注入"""
    create = MemoryUpdateResult(operations=[
        MemoryOp(op="create", key="user_role", content="用户是经理", category="role")])
    model = _mock_model("direct", task="闲聊", memory_result=create)
    rag_chain_query(db_session, "我是经理", "sk-test", model=model, conversation_id="A")
    # B 会话：新模型避免 A 的 mock 计数干扰，仅验证注入
    model_b = _mock_model("direct", task="闲聊")
    rag_chain_query(db_session, "你好", "sk-test", model=model_b, conversation_id="B")
    assert "用户是经理" in str(model_b.rw_mock.call_args)


def test_manual_delete_not_injected():
    put_memory(get_memory_store(), "k1", "临时偏好", "preference")
    delete_memory(get_memory_store(), "k1")
    model = _mock_model("direct", task="闲聊")
    rag_chain_query(None, "你好", "sk-test", model=model)
    assert "临时偏好" not in str(model.rw_mock.call_args)


def test_extraction_runs_after_answer_and_writes_store():
    """问答收口触发提取：mem_mock 被调用且记忆落 Store"""
    create = MemoryUpdateResult(operations=[
        MemoryOp(op="create", key="answer_style", content="回答要简洁", category="instruction")])
    model = _mock_model("direct", task="闲聊", memory_result=create)
    resp = rag_chain_query(None, "以后简洁点", "sk-test", model=model)
    assert resp.response_type == "answer"
    model.mem_mock.assert_called()
    assert any(m["key"] == "answer_style" for m in read_memories(get_memory_store()))
