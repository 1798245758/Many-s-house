"""错误诊断 Agent 上下文管理测试

模拟用户对同一个失败任务连续多轮追问，对比两种策略：
  - 混乱策略（naive）：每轮把全部历史问答 + 完整证据原文重复追加，上下文线性膨胀，
    证据被反复复制，早期关键信息被后期噪声淹没
  - 管理策略（ContextManager）：证据只钉一次，滑动窗口限轮数，超预算滚动摘要压缩，
    上下文规模恒定有界

断言覆盖：证据去重、窗口有界、压缩触发、会话隔离、跨任务不污染、降级容错、API 联通。
"""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.database import Base, get_db
from app.models.setting import Setting
from app.services.diagnosis import context as ctx


class FakeClient:
    """记录每轮收到的 messages，返回可预测答案；用于观测上下文规模"""

    def __init__(self, answer_prefix="根因"):
        self.calls = []
        self.answer_prefix = answer_prefix

    def messages(self, messages, temperature=0.3):
        self.calls.append(messages)
        n = len(self.calls)
        return f"{self.answer_prefix}分析第{n}轮：基于证据判断为文件损坏导致解析失败。"

    def ask(self, system_prompt, user_message, temperature=0.3):
        # 供滚动摘要压缩调用：返回合并后的摘要
        self.calls.append([{"role": "system", "content": user_message}])
        return "【会话摘要】任务t1失败于extract阶段，根因文件损坏，用户关注修复步骤。"

    @staticmethod
    def total_chars(messages):
        return sum(len(m["content"]) for m in messages)


# 一份"大体积"证据，用于暴露重复堆积问题
@pytest.fixture
def big_evidence():
    return {
        "task_id": "t1",
        "task_type": "doc_ingest",
        "final_status": "failed",
        "final_stage": "extract",
        "error": "PDF 解析失败：文件损坏，页对象流缺失 EOF 标记",
        "retry_count": 2,
        "events": [{"status": "running", "stage": "extract", "message": f"解析第{i}页"} for i in range(40)],
        "document": {"id": 1, "filename": "坏文件.pdf", "file_type": "pdf",
                     "file_size": 204800, "status": "error"},
        "error_log": "Traceback ...\n" + "X" * 2000,
    }


# ---------- 混乱策略（对照组）：证明"未管理"会失控 ----------

def _naive_accumulate(history, evidence_text, question):
    """朴素做法：每轮把完整证据 + 全部历史原文再拼一遍"""
    parts = [evidence_text]
    for q, a in history:
        parts.append(f"问:{q}\n答:{a}")
    parts.append(f"问:{question}")
    return "\n".join(parts)


def test_naive_context_explodes(big_evidence):
    evidence_text = ctx._evidence_to_text(big_evidence)
    history = []
    prompts = []
    for i in range(8):
        prompt = _naive_accumulate(history, evidence_text, f"第{i}个问题：为什么失败？")
        prompts.append(prompt)
        history.append((f"第{i}个问题：为什么失败？", "根因是文件损坏。" * 20))
    sizes = [len(p) for p in prompts]
    # 混乱特征一：上下文随轮数单调无界增长
    assert sizes == sorted(sizes) and sizes[-1] > sizes[0] + 1000
    # 混乱特征二：完整证据原文在每一轮都被重复携带（8 轮共 8 次）
    assert sum(p.count("状态机流转事件日志") for p in prompts) == 8
    # 混乱特征三：早期问题被后期噪声逐渐淹没（历史区占比越来越高）
    history_ratio_0 = 1 - len(evidence_text) / sizes[0]
    history_ratio_last = 1 - len(evidence_text) / sizes[-1]
    assert history_ratio_last > history_ratio_0


# ---------- 管理策略：证据钉住 + 滑动窗口 + 滚动摘要 ----------

def test_evidence_pinned_once(big_evidence):
    """证据只钉在头部一次，不随追问轮数重复"""
    client = FakeClient()
    mgr = ctx.ContextManager("t1", big_evidence, client)
    for i in range(6):
        messages, _ = mgr.build_messages(f"第{i}问：失败原因？")
        mgr.add_turn(f"第{i}问：失败原因？", client.messages(messages))
    # 最后一轮：证据文本在整段消息里只出现一次
    evidence_marker = "任务现场证据"
    joined = "\n".join(m["content"] for m in messages)
    assert joined.count(evidence_marker) == 1


def test_sliding_window_bounds_context(big_evidence):
    """上下文规模有界：追问 10 轮后仍不超过预算太多"""
    client = FakeClient()
    mgr = ctx.ContextManager("t1", big_evidence, client)
    sizes = []
    for i in range(10):
        messages, stats = mgr.build_messages(f"第{i}问：还需要什么信息？")
        sizes.append(stats["context_chars"])
        mgr.add_turn(f"第{i}问：还需要什么信息？", client.messages(messages))
    # 触发过压缩后，上下文不会无限膨胀
    assert max(sizes) <= ctx.MAX_CONTEXT_CHARS + ctx.EVIDENCE_MAX_CHARS + 4000
    assert mgr.compact_count >= 1


def test_window_keeps_only_recent_turns(big_evidence):
    """窗口内只保留最近 N 轮完整原文，更早的进摘要"""
    client = FakeClient()
    mgr = ctx.ContextManager("t1", big_evidence, client)
    for i in range(8):
        messages, _ = mgr.build_messages(f"Q{i}")
        mgr.add_turn(f"Q{i}", f"A{i}")
    assert len(mgr.turns) <= ctx.RECENT_TURNS
    assert mgr.summary  # 旧轮次已折叠进摘要


def test_summary_fallback_when_llm_fails(big_evidence):
    """LLM 压缩失败时降级规则抽取，摘要仍非空"""
    client = FakeClient()
    client.ask = MagicMock(side_effect=RuntimeError("summarize boom"))
    mgr = ctx.ContextManager("t1", big_evidence, client)
    for i in range(8):
        messages, _ = mgr.build_messages(f"Q{i}")
        mgr.add_turn(f"Q{i}", f"A{i}")
    assert mgr.summary
    assert mgr.compact_count >= 1


def test_clip_folds_long_text():
    long = "A" * 5000
    clipped = ctx._clip(long, 1000)
    assert len(clipped) < 1200
    assert "已折叠" in clipped


# ---------- 会话隔离与跨任务防污染 ----------

def test_ask_creates_and_reuses_session(big_evidence):
    ctx.clear_sessions()
    client = FakeClient()
    r1 = ctx.ask("t1", "为什么失败？", big_evidence, client)
    sid = r1["session_id"]
    assert sid and r1["answer"]
    assert r1["context_stats"]["total_turns"] == 1
    # 复用会话：evidence=None 命中缓存，不重新建会话
    r2 = ctx.ask("t1", "怎么修复？", None, client, session_id=sid)
    assert r2["session_id"] == sid
    assert r2["context_stats"]["total_turns"] == 2


def test_ask_switches_task_creates_new_session(big_evidence):
    ctx.clear_sessions()
    client = FakeClient()
    r1 = ctx.ask("t1", "为什么失败？", big_evidence, client)
    sid = r1["session_id"]
    other = dict(big_evidence, task_id="t2")
    # 同一 session_id 但换了 task_id → 强制新建会话，避免跨任务上下文污染
    r2 = ctx.ask("t2", "这个任务呢？", other, client, session_id=sid)
    assert r2["session_id"] != sid


def test_ask_new_session_requires_evidence():
    ctx.clear_sessions()
    with pytest.raises(ValueError):
        ctx.ask("t1", "问", None, FakeClient())


def test_llm_error_degrades_gracefully(big_evidence):
    ctx.clear_sessions()
    client = FakeClient()
    client.messages = MagicMock(side_effect=RuntimeError("timeout"))
    r = ctx.ask("t1", "为什么失败？", big_evidence, client)
    assert r["answer"] == ""
    assert "timeout" in r["llm_error"]


# ---------- API 联通 ----------

@pytest.fixture
def client_api():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app), Session
    app.dependency_overrides.clear()


def test_chat_api_requires_manager(client_api):
    c, _ = client_api
    body = c.post("/api/diagnosis/chat",
                  json={"task_id": "t1", "question": "为什么失败"},
                  headers={"X-Role": "employee"}).json()
    assert body["code"] == "PERMISSION_DENIED"


def test_chat_api_multi_turn_flow(client_api):
    c, Session = client_api
    ctx.clear_sessions()
    db = Session()
    db.add(Setting(key="api_key", value="sk-test"))
    db.commit()
    db.close()

    ev = {"task_id": "t1", "task_type": "doc_ingest", "final_status": "failed",
          "final_stage": "extract", "error": "文件损坏", "retry_count": 0,
          "events": [], "document": None, "error_log": ""}
    with patch("app.services.tasks.runner.get_task_snapshot",
               return_value={"status": "failed", "stage": "extract", "error": "文件损坏", "doc_id": None}), \
         patch("app.services.tasks.runner.get_task_history", return_value=[]), \
         patch("app.services.deepseek.DeepSeekClient.messages", return_value="根因：文件损坏"):
        r1 = c.post("/api/diagnosis/chat",
                    json={"task_id": "t1", "question": "为什么失败？"},
                    headers={"X-Role": "manager"}).json()
    assert r1["code"] == "SUCCESS"
    sid = r1["data"]["session_id"]
    assert r1["data"]["answer"] == "根因：文件损坏"

    # 第二轮携带 session_id：复用会话，不再采集证据
    with patch("app.services.tasks.runner.get_task_snapshot") as snap, \
         patch("app.services.deepseek.DeepSeekClient.messages", return_value="修复：重新导出PDF"):
        r2 = c.post("/api/diagnosis/chat",
                    json={"task_id": "t1", "question": "怎么修复？", "session_id": sid},
                    headers={"X-Role": "manager"}).json()
    assert r2["code"] == "SUCCESS"
    assert r2["data"]["session_id"] == sid
    assert r2["data"]["context_stats"]["total_turns"] == 2
    snap.assert_not_called()  # 复用会话不重新采集 checkpoint


def test_chat_api_not_found(client_api):
    c, Session = client_api
    db = Session()
    db.add(Setting(key="api_key", value="sk-test"))
    db.commit()
    db.close()
    with patch("app.services.tasks.runner.get_task_snapshot", return_value=None):
        body = c.post("/api/diagnosis/chat",
                      json={"task_id": "nope", "question": "为什么失败"},
                      headers={"X-Role": "manager"}).json()
    assert body["code"] == "NOT_FOUND"
