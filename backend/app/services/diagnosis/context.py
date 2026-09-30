"""错误诊断 Agent 的上下文管理（参考 Claude Code 的上下文工程实践）

多层上下文模型，按稳定性分层组装，每轮请求重建而非无限追加：
  L0 系统规则层   SYSTEM_PROMPT，恒定不变（可被 LLM 缓存）
  L1 证据钉住层   现场证据只在会话创建时采集一次，压缩后"钉"在上下文头部，
                 永不被对话轮次挤出（对应 Claude Code 的 system prompt 钉住文件上下文）
  L2 滚动摘要层   超出窗口的旧轮次被压缩进 running summary，只保留关键事实
  L3 近期窗口层   最近 N 轮完整问答原文

防混乱机制：
  - 滑动窗口：只保留最近 RECENT_TURNS 轮原文，上下文有硬预算 MAX_CONTEXT_CHARS
  - 自动压缩（compaction）：超预算时旧轮次折叠为摘要，LLM 压缩失败降级规则抽取
  - 证据去重：证据/错误日志每会话只采集一次，不随提问重复堆积
  - 会话隔离：每个 session_id 独立上下文，互不污染；LRU 淘汰防内存泄漏
  - 可观测性：每轮返回 context_stats（各层大小、是否触发压缩、压缩比）
"""
import json
import logging
import threading
from collections import OrderedDict

from app.services.diagnosis.agent import (
    SYSTEM_PROMPT, RETRIEVAL_SYSTEM_PROMPT, _evidence_to_text, _query_evidence_to_text)

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 12000    # L1+L2+L3 总预算（不含系统规则层）
EVIDENCE_MAX_CHARS = 5000    # 钉住证据的压缩上限
RECENT_TURNS = 3             # 完整保留的近期轮数
SUMMARY_MAX_CHARS = 1500     # 滚动摘要上限
ANSWER_MAX_CHARS = 2000      # 单轮答案入库上限（防单轮超长挤爆窗口）
_MAX_SESSIONS = 64           # 会话 LRU 上限

SUMMARIZE_PROMPT = (
    "你在维护一次错误诊断会话的上下文摘要。请把已有摘要与新的对话轮次合并压缩，"
    "只保留对后续诊断有用的关键事实：任务/文档标识、错误信息与阶段、已确认的根因结论、"
    "已给出的修复建议要点、用户持续关注的方向、已被排除的假设。"
    "丢弃寒暄、重复内容和完整证据原文（证据已单独钉住，无需进摘要）。"
    f"直接输出合并后的摘要正文，不超过 {SUMMARY_MAX_CHARS} 字。\n\n"
    "已有摘要:\n{summary}\n\n新增对话轮次:\n{turns}"
)


def _clip(text: str, limit: int, head_ratio: float = 0.6) -> str:
    """超长截断：保头保尾（错误日志的结论常在尾部），中间以省略标记折叠"""
    if not text or len(text) <= limit:
        return text or ""
    head = int(limit * head_ratio)
    tail = limit - head - 16
    return f"{text[:head]}\n…[已折叠 {len(text) - limit} 字]…\n{text[-tail:]}"


def _rule_summarize(turns: list) -> str:
    """LLM 压缩失败时的降级：规则抽取每轮问题原文 + 答案首段要点"""
    lines = []
    for t in turns:
        answer_brief = t["answer"].split("\n\n")[0][:120] if t["answer"] else "（无答案）"
        lines.append(f"- 问: {t['question'][:80]} → 答要点: {answer_brief}")
    return "\n".join(lines)[-SUMMARY_MAX_CHARS:]


class ContextManager:
    """单会话上下文管理器：证据钉住 + 滑动窗口 + 滚动摘要"""

    def __init__(self, subject_id, evidence: dict, client=None, subject_type: str = "task"):
        self.subject_id = subject_id
        self.subject_type = subject_type
        self.client = client  # 用于滚动摘要压缩；None 时始终走规则降级
        self.system_prompt = RETRIEVAL_SYSTEM_PROMPT if subject_type == "query" else SYSTEM_PROMPT
        text_fn = _query_evidence_to_text if subject_type == "query" else _evidence_to_text
        self.pinned_evidence = _clip(text_fn(evidence), EVIDENCE_MAX_CHARS)
        self.turns: list = []        # 仅保留窗口内（未折叠）的近期轮次
        self.total_turns = 0         # 累计轮数（含已折叠进摘要的）
        self.summary = ""
        self.compact_count = 0
        self._lock = threading.Lock()

    @property
    def task_id(self):
        """向后兼容别名：既有 router/测试按 task_id 读取主体标识"""
        return self.subject_id

    def add_turn(self, question: str, answer: str):
        """入库一轮问答后立即维护窗口不变式（超出则压缩），使 turns 始终有界。
        与 build_messages 共用同一把锁：FastAPI 同步端点跑线程池，同会话并发追问需互斥。"""
        with self._lock:
            self.turns.append({
                "question": _clip(question, 500),
                "answer": _clip(answer, ANSWER_MAX_CHARS),
            })
            self.total_turns += 1
            if self._need_compact():
                self._compact()

    def _window_chars(self) -> int:
        return sum(len(t["question"]) + len(t["answer"]) for t in self.turns)

    def _need_compact(self) -> bool:
        # 轮数溢出窗口，或（钉住证据+摘要+窗口）超预算
        if len(self.turns) > RECENT_TURNS:
            return True
        return (len(self.pinned_evidence) + len(self.summary) + self._window_chars()
                > MAX_CONTEXT_CHARS)

    def _compact(self):
        """压缩：窗口外旧轮次折叠进滚动摘要（Claude Code compaction 思路）"""
        if len(self.turns) > RECENT_TURNS:
            overflow = self.turns[:-RECENT_TURNS]
            turns_text = "\n".join(f"问: {t['question']}\n答: {t['answer']}" for t in overflow)
            prompt = SUMMARIZE_PROMPT.format(summary=self.summary or "（无）", turns=turns_text)
            if self.client is not None:
                try:
                    merged = self.client.ask(self.system_prompt, prompt)
                    self.summary = _clip(merged.strip(), SUMMARY_MAX_CHARS)
                except Exception as e:
                    logger.warning(f"摘要压缩 LLM 调用失败，降级规则抽取: {e}")
                    self.summary = _clip(self.summary + "\n" + _rule_summarize(overflow),
                                         SUMMARY_MAX_CHARS)
            else:
                self.summary = _clip(self.summary + "\n" + _rule_summarize(overflow),
                                     SUMMARY_MAX_CHARS)
            del self.turns[:-RECENT_TURNS]
        else:
            # 窗口内已无旧轮可折叠，只能收紧近期轮次的答案以压回预算
            for t in self.turns:
                t["answer"] = _clip(t["answer"], ANSWER_MAX_CHARS // 2)
        self.compact_count += 1

    def build_messages(self, question: str) -> tuple:
        """组装本轮 LLM 消息，返回 (messages, stats)。窗口不变式已由 add_turn 维护。"""
        with self._lock:
            recent = self.turns[-RECENT_TURNS:]
            label = "检索现场证据" if self.subject_type == "query" else "任务现场证据"
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"【{label}（已固定，勿要求重复提供）】\n{self.pinned_evidence}"},
                {"role": "assistant", "content": "已收到现场证据，后续所有分析都将基于这份证据。请提问。"},
            ]
            if self.summary:
                messages.append({"role": "system",
                                 "content": f"【更早对话的压缩摘要】\n{self.summary}"})
            for t in recent:
                messages.append({"role": "user", "content": t["question"]})
                messages.append({"role": "assistant", "content": t["answer"]})
            messages.append({"role": "user", "content": question})
            ctx_size = sum(len(m["content"]) for m in messages)
            stats = {
                "total_turns": self.total_turns + 1,
                "window_turns": len(recent) + 1,
                "summary_chars": len(self.summary),
                "evidence_chars": len(self.pinned_evidence),
                "context_chars": ctx_size,
                "budget": MAX_CONTEXT_CHARS,
                "compact_count": self.compact_count,
            }
            return messages, stats


class _SessionStore:
    """进程内会话存储（LRU 淘汰），与短期记忆"退出即清理"语义一致"""

    def __init__(self):
        self._sessions = OrderedDict()
        self._lock = threading.Lock()

    def get(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                self._sessions.move_to_end(session_id)
                return self._sessions[session_id]
        return None

    def create(self, session_id: str, manager: ContextManager) -> ContextManager:
        with self._lock:
            self._sessions[session_id] = manager
            while len(self._sessions) > _MAX_SESSIONS:
                self._sessions.popitem(last=False)
        return manager

    def clear(self):
        with self._lock:
            self._sessions.clear()


_sessions = _SessionStore()
clear_sessions = _sessions.clear


def ask(subject_id, question: str, evidence: dict | None, client,
        session_id: str | None = None, subject_type: str = "task") -> dict:
    """多轮诊断追问入口：首轮建会话钉证据，后续轮走上下文管理。

    evidence 仅在新建会话（或切换主体）时必传；命中已有会话时可为 None。
    返回 {session_id, task_id, answer, context_stats, llm_error}
    """
    manager = _sessions.get(session_id) if session_id else None
    if (manager is None or manager.subject_id != subject_id
            or manager.subject_type != subject_type):
        # 新会话（或换了主体）：重新采集钉住证据，避免跨主体上下文污染
        if evidence is None:
            raise ValueError("新建诊断会话必须提供 evidence")
        import uuid
        session_id = uuid.uuid4().hex
        manager = _sessions.create(
            session_id, ContextManager(subject_id, evidence, client, subject_type))
    manager.client = client  # 刷新请求级 client（api_key 可能轮换）

    messages, stats = manager.build_messages(question)
    try:
        answer = client.messages(messages)
        llm_error = ""
    except Exception as e:
        answer = ""
        llm_error = f"LLM 诊断调用失败: {e}"
    manager.add_turn(question, answer)
    return {
        "session_id": session_id,
        "task_id": subject_id,
        "answer": answer,
        "context_stats": stats,
        "llm_error": llm_error,
    }
