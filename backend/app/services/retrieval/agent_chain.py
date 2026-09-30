"""基于 LangGraph 状态图的知识库问答 Graph

===== 图结构 =====
START → permission_gate（确定性门禁，不调 LLM）
          ├─ 员工命中经理专属主题 → END（permission_denied）
          └─ 放行 → intent（意图关卡子图：classify + slot，基于原始问题）
intent 条件边按 action 分流：
          ├─ refuse  → END（固定拒绝话术，跳过改写省一次 LLM 调用）
          ├─ clarify → END（澄清问题 + 意图分析，跳过改写）
          └─ direct/search → rewrite（Query 理解：改写 + 分解 + 关键词优化，
                              并合成首跳检索串）
                rewrite 条件边按 action 分流：
                ├─ direct → direct 节点（空上下文直答）→ END
                └─ search → search_plan → embed → search（循环）
                        route_search：队列非空且未达 HOP_MAX_HOPS
                          → hop_reason（LLM 依赖推理下一跳）→ 回到 embed
                          否则 → rerank（累积候选统一重排）
                        ├─ 无命中 → END（固定话术，不调 LLM）
                        └─ 有命中 → verify（四条件证据校验）
                              ├─ 通过/硬停 → context → answer → END
                              └─ 不满足且可继续 → 回 embed 补充检索

检索与生成分离在条件边上：rerank 后若无命中直接结束，
避免无命中时白白消耗一次 LLM 调用、或 LLM 异常时吞掉固定兜底话术。
短期记忆：编译时挂 MemorySaver Checkpointer（状态存进程内存，程序退出
自动清理，零落库）；conversation_id 当 thread_id，跨轮恢复近期问答与任务状态。
"""
import json
import math
import struct
import logging
from datetime import datetime
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from sqlalchemy.orm import Session

from app.services.deepseek import DeepSeekClient
from app.services.retrieval.searcher import hybrid_search
from app.services.retrieval.rewrite_chain import build_query_rewriter
from app.services.retrieval.intent_chain import build_intent_gate
from app.services.retrieval.hop_chain import build_hop_reasoner
from app.services.retrieval.verify_chain import build_evidence_verifier
from app.services.retrieval.page_tools import read_page
from app.services.retrieval.memory_store import (
    get_memory_store, read_memories, format_memories)
from app.services.retrieval.memory_chain import (
    build_memory_extractor, apply_memory_ops)
from app.services.embedding import embed_text, EMBEDDING_DIM
from app.services import vector_store
from app.services.role import ROLE_MANAGER, manager_doc_ids
from app.config import (MANAGER_KEYWORDS, HOP_MAX_HOPS, HOP_TOP_K_PER_HOP,
                        HOP_TOP_K_CONTEXT, VERIFY_MISSING_STREAK,
                        PAGE_READ_ENABLED, PAGE_READ_MAX_PAGES, PAGE_READ_MAX_CHARS)
from app.schemas.query import SourceInfo, QueryResponse, IntentInfo
from app.models.query_trace import QueryTrace

logger = logging.getLogger(__name__)

# 系统提示：企业场景下有上下文只依据上下文作答；无上下文时闲聊可自然回应
SYSTEM_PROMPT = (
    "你是'企业问答助手'，一个企业知识库助手，知识库内容为企业内部管理资料（企业文化、员工手册、管理制度、培训体系等）。"
    "若提供了上下文，只依据上下文回答问题，回答需专业、准确，"
    "并在回答末尾用【来源】一行列出引用的资料名（去重）；"
    "上下文没有相关信息时，诚实回答'根据现有资料无法回答此问题'，不要编造。"
    "不得给出资料内容之外的臆测结论。"
    "若没有上下文且问题属于闲聊问候，请自然、简短地回应。"
)

# 意图关卡命中安全风险时的固定拒绝话术（不再调用 LLM）
REFUSAL_MESSAGE = "抱歉，我无法回答该问题。请提出与企业知识库相关的问题。"

# 员工提问命中经理专属主题时的固定话术（不再调用 LLM）
PERMISSION_DENIED_MESSAGE = "该问题涉及经理专属内容，当前员工权限不足，无法提供相关信息。"

# search 无命中时的固定话术（不再调用 LLM）
NO_RESULT_MESSAGE = "当前知识库中没有相关信息，请先上传文档。"

# 检索轨迹三级分级：正常 / 降级警告 / 错误；轮级取各步最高级
_LEVEL_ORDER = {"normal": 0, "degraded": 1, "error": 2}


def _step(step: str, level: str, code: str = "", message: str = "", **metrics) -> dict:
    """构造一条纯数据步记录（可 msgpack 序列化，进得了 checkpoint）"""
    return {"step": step, "level": level, "code": code, "message": message,
            "metrics": metrics, "ts": datetime.now().isoformat(timespec="seconds")}


def _classify_trace(trace: list) -> tuple[str, str]:
    """轮级严重度 = 各步最高级（同级取最后一个）。返回 (final_level, final_code)。"""
    best, best_code = "normal", ""
    for s in trace or []:
        if _LEVEL_ORDER.get(s.get("level"), 0) >= _LEVEL_ORDER[best]:
            best = s.get("level", "normal")
            best_code = s.get("code", "")
    return best, best_code


# 编译图进程级单例缓存：键为 (api_key, id(model))（Key 变更自动重建；
# 测试注入 mock model 时同 mock 复用，共享同一 MemorySaver 以支持多轮）。
# 编译图本身无状态，每次 invoke 的 state 独立，可安全跨请求复用；
# 请求级 db 不经闭包捕获，改由 invoke 的 RunnableConfig 注入
_GRAPH_CACHE: dict = {}

# 短期记忆：每会话保留最近 HISTORY_TURNS 轮问答条目，
# 条目内答案截断 _ANSWER_SNIPPET_LEN 字控制 prompt 增量
HISTORY_TURNS = 3
_ANSWER_SNIPPET_LEN = 300


def _keep_recent_history(old: list, new: list) -> list:
    """历史 reducer：追加入新条目并只保留最近 HISTORY_TURNS 条"""
    return ((old or []) + (new or []))[-HISTORY_TURNS:]


def _build_model(api_key: str):
    """init_chat_model：统一模型初始化入口，走 OpenAI 兼容协议接入 DeepSeek"""
    return init_chat_model(
        "deepseek-chat",
        model_provider="openai",
        base_url="https://api.deepseek.com",
        api_key=api_key,
        temperature=0.7,
    )


def embed_query(text: str) -> list[float]:
    """查询文本 → BGE 向量（与旧 generator 中逻辑一致）"""
    packed = embed_text(text)
    return list(struct.unpack(f"{EMBEDDING_DIM}f", packed))


def _cosine(a: list[float], b: list[float]) -> float:
    """余弦相似度（精排打分用）"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _to_intent_info(slots) -> IntentInfo:
    """SlotResult → IntentInfo（空串归一为 None）"""
    return IntentInfo(
        task=slots.task or "未知",
        entities=list(slots.entities or []),
        time=slots.time or None,
        risk_note=slots.risk_note or None,
    )


# ===== 1. 定义状态：在节点间流转的共享数据结构 =====
class RAGState(TypedDict, total=False):
    question: str                # 用户原始问题（门禁/历史/展示用）
    rewritten_question: str      # 改写后的规范问题（直答/回答用）
    sub_questions: list          # 复合问题分解出的子问题（仅日志展示）
    rewrite_keywords: list       # 改写阶段优化出的检索关键词
    role: str                    # employee | manager
    permission_denied: bool      # 角色主题门禁命中
    action: str                  # refuse | clarify | direct | search
    slots: object                # SlotResult | None（意图关卡槽位）
    search_query: str            # 首跳检索关键词串（改写后合并改写/槽位关键词，原问题兜底）
    current_query: str           # 当前跳检索串（首跳后由多跳推理更新）
    query_queue: list            # 剩余子问题队列（多跳驱动序列）
    loop_finished: bool          # 多跳推理已判定终止（本跳查询仍检索，之后收尾）
    hop_count: int               # 已执行检索跳数（上限 HOP_MAX_HOPS）
    all_hits: list               # 各跳累积候选（纯数据 dict：id/content/document_name，
                                 # 按 chunk_id 去重；ORM 对象不可序列化，进不了 checkpoint）
    no_new_hits: bool            # 上一轮检索零新增候选（硬停条件之一）
    verify_retrying: bool        # verify 本轮判定可继续补检（未写 evidence_status，
                                 # 避免上一轮残留值让 route_verify 误走 context）
    embedding: list[float]       # 查询向量
    ranked: list                 # 余弦相似度重排后的候选（同 all_hits 的纯数据 dict）
    evidence_status: str         # verified | uncertain（证据校验结论）
    missing_aspects: list        # 硬停时的缺失方面（注入对冲 prompt）
    missing_streak: int          # 连续相同缺失方面的轮数（硬停条件之一）
    last_missing: list           # 上一轮校验的缺失方面（排序后）
    context: str                 # 带来源标注的上下文
    answer: str                  # 最终回答文本
    conversation_history: Annotated[list, _keep_recent_history]  # 短期记忆：近期问答条目
    last_task_status: str        # 短期记忆：上一轮任务完成状态
    trace: list                  # 本轮多步诊断轨迹（普通字段：节点覆盖式累积，入口重置）


def build_rag_graph(api_key: str, model=None):
    """构建问答状态图并编译返回

    节点不捕获任何请求级资源：db 由每次 invoke 经 RunnableConfig 的
    configurable 注入，因此编译图可作为进程级单例跨请求复用
    （见 get_compiled_graph），省去每请求重建开销。
    model 参数仅供测试注入 mock。
    """
    if model is None:
        model = _build_model(api_key)
    parser = StrOutputParser()

    # ===== 2. 定义节点：每个节点是一个函数，读 State、写 State =====
    def permission_gate_node(state: RAGState) -> dict:
        """角色主题门禁：员工提问命中经理专属关键词直接短路，不调 LLM。
        作为图入口，同时重置本轮 trace（MemorySaver 会带回上轮轨迹）"""
        if state["role"] != ROLE_MANAGER and any(
            kw in state["question"] for kw in MANAGER_KEYWORDS
        ):
            return {"permission_denied": True, "answer": PERMISSION_DENIED_MESSAGE,
                    "trace": [_step("permission_gate", "normal", code="N2", message="权限拦截")]}
        return {"permission_denied": False, "trace": []}

    def intent_node(state: RAGState) -> dict:
        """意图关卡（classify + slot 子图），基于原始问题分类，与改写阶段解耦：
        不消费改写产出，拒绝/澄清短路时还能省下改写的一次 LLM 调用。
        首跳检索串的关键词合成放在 rewrite 节点（两者产出先后都合时）。

        门禁短路路径不会到达本节点，故在此惰性构建关卡，
        避免被拦截的请求白白初始化结构化输出链。
        """
        gate = build_intent_gate(api_key, model=model)
        intent = gate(state["question"])
        action = intent["action"]
        code = {"refuse": "N3", "clarify": "N4"}.get(action, "")
        trace = state.get("trace", []) + [
            _step("intent", "normal", code=code, message=f"action={action}")]
        if intent.get("degraded"):
            trace = trace + [_step("intent", "degraded", code="W3", message="意图识别降级")]
        return {"action": action, "slots": intent["slots"], "trace": trace}

    def rewrite_node(state: RAGState) -> dict:
        """Query 理解：改写 + 分解 + 关键词优化（单次结构化调用），
        并把改写关键词与意图槽位关键词合并去重为首跳检索串（数据仅由意图单向
        流入改写，改写不反向影响意图判定）

        门禁/拒绝/澄清短路路径不会到达本节点，故在此惰性构建，
        避免被拦截的请求白白初始化结构化输出链。
        """
        rewriter = build_query_rewriter(api_key, model=model)
        result = rewriter(state["question"],
                          history=state.get("conversation_history"),
                          task_status=state.get("last_task_status", ""),
                          memories=format_memories(read_memories(get_memory_store())))
        slots = state.get("slots")
        # 改写关键词 + 槽位关键词合并去重；子问题关键词已由改写阶段覆盖，
        # 仍走单次混合检索；两者皆空时兜底原问题
        merged = list(dict.fromkeys(
            (result.keywords or [])
            + (slots.search_keywords if slots is not None else [])
        ))
        search_query = " ".join(merged) if merged else state["question"]
        out = {
            "rewritten_question": result.rewritten_question,
            "sub_questions": result.sub_questions,
            "rewrite_keywords": result.keywords,
            "search_query": search_query,
        }
        if getattr(result, "degraded", False):
            out["trace"] = state.get("trace", []) + [
                _step("rewrite", "degraded", code="W2", message="改写降级")]
        return out

    def direct_node(state: RAGState) -> dict:
        """direct 分支：空上下文直答（注入长期记忆约束身份/格式/偏好）"""
        memories = format_memories(read_memories(get_memory_store()))
        direct_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "长期记忆（用户身份/偏好，回答需遵守）:\n{memories}\n\n问题: {question}"),
            ])
            | model
            | parser
        )
        answer = direct_chain.invoke(
            {"question": state["rewritten_question"], "memories": memories})
        return {"answer": answer,
                "trace": state.get("trace", []) + [
                    _step("answer", "normal", code="N5", message="直答闲聊")]}

    def search_plan_node(state: RAGState) -> dict:
        """初始化多跳循环：首跳查询 + 子问题队列（无子问题时队列为空，
        首跳后直接重排，行为与单跳一致）。
        同时重置上一轮写入的循环结论字段：checkpointer 恢复会带回上轮
        终态，不重置会让第二轮的多跳/补检循环被跳过"""
        return {
            "current_query": state["search_query"],
            "query_queue": list(state.get("sub_questions") or []),
            "hop_count": 0,
            "all_hits": [],
            "no_new_hits": False,
            "loop_finished": False,
            "verify_retrying": False,
            "missing_streak": 0,
            "last_missing": [],
            "evidence_status": "",
            "missing_aspects": [],
        }

    def embed_node(state: RAGState) -> dict:
        """当前跳检索串 → 768维查询向量"""
        return {"embedding": embed_query(state["current_query"])}

    def search_node(state: RAGState, config: RunnableConfig) -> dict:
        """当前跳混合检索（员工角色排除经理专属文档），按 chunk_id 去重累积。
        db 从运行时 config 读取（单例图不能闭包捕获请求级 Session）；
        ORM Chunk 立即转纯数据 dict——checkpoint 序列化要求状态可 msgpack，
        且跨请求 Session 的 ORM 对象本身也不安全"""
        db = config["configurable"]["db"]
        exclude = manager_doc_ids(db) if state["role"] != ROLE_MANAGER else None
        hits = hybrid_search(db, state["embedding"], state["current_query"],
                             top_k=HOP_TOP_K_PER_HOP, exclude_doc_ids=exclude)
        seen = {c["id"] for c in state.get("all_hits", [])}
        new_hits = [{"id": c.id, "content": c.content,
                     "document_name": c.document.filename if c.document else "未知",
                     "document_id": c.document_id,
                     "page": c.page}
                    for c in hits if c.id not in seen]
        all_hits = state.get("all_hits", []) + new_hits
        return {"all_hits": all_hits,
                "hop_count": state.get("hop_count", 0) + 1,
                "no_new_hits": not new_hits,
                "trace": state.get("trace", []) + [
                    _step("search", "normal", message="混合检索",
                          new_hits=len(new_hits), total=len(all_hits))]}

    def hop_reason_node(state: RAGState) -> dict:
        """多跳推理：证据已足则清空队列终止，否则生成下一跳查询并消费队首

        证据摘要取累积候选 Top3 内容截断，控制推理入参长度。
        """
        reasoner = build_hop_reasoner(api_key, model=model)
        evidence = "\n".join(
            f"[来源: {c['document_name']}]\n{c['content'][:200]}"
            for c in state["all_hits"][:3]
        )
        decision = reasoner(state["rewritten_question"], evidence, state["query_queue"])
        trace = state.get("trace", [])
        if getattr(decision, "degraded", False):
            trace = trace + [_step("hop_reason", "degraded", code="W4", message="多跳推理降级")]
        if decision.finished:
            return {"query_queue": [], "loop_finished": True, "trace": trace}
        next_query = (" ".join(decision.next_keywords)
                      if decision.next_keywords else decision.next_query)
        return {"current_query": next_query or state["query_queue"][0],
                "query_queue": state["query_queue"][1:], "loop_finished": False,
                "trace": trace}

    def rerank_node(state: RAGState) -> dict:
        """多跳累积候选统一余弦重排，截断保留 HOP_TOP_K_CONTEXT 个"""
        hits = state["all_hits"]
        if len(hits) <= 1:
            ranked = hits
        else:
            embeddings = vector_store.get_embeddings([c["id"] for c in hits])

            def score(chunk):
                vec = embeddings.get(chunk["id"])
                # ChromaDB可能返回numpy数组，不能用if vec判空；缺失时记-1排最后
                return _cosine(state["embedding"], list(vec)) if vec is not None else -1.0

            ranked = sorted(hits, key=score, reverse=True)[:HOP_TOP_K_CONTEXT]
        step = (_step("rerank", "error", code="E1", message="候选集为空", ranked=0)
                if not ranked else
                _step("rerank", "normal", message="精排", ranked=len(ranked)))
        return {"ranked": ranked, "trace": state.get("trace", []) + [step]}

    def verify_node(state: RAGState) -> dict:
        """证据校验：四条件全过 → verified；否则查硬停条件，
        未硬停则写入换角度的补充查询继续循环"""
        verifier = build_evidence_verifier(api_key, model=model)
        evidence = "\n".join(
            f"[来源: {c['document_name']}]\n{c['content'][:200]}"
            for c in state["ranked"]
        )
        result = verifier(state["rewritten_question"],
                          state.get("sub_questions") or [], evidence)
        trace = state.get("trace", [])
        if getattr(result, "degraded", False):
            trace = trace + [_step("verify", "degraded", code="W5", message="证据校验降级")]
        at_hop_limit = state["hop_count"] >= HOP_MAX_HOPS
        if result.sufficient:
            # 校验充分即正常好结果（N1 由 answer 节点记录）；W6 仅用于“达上限仍不足”
            return {"evidence_status": "verified", "verify_retrying": False, "trace": trace}
        # 硬停三条件：达检索轮数上限 / 缺失方面连续相同 / 上一轮零新增候选（无更多知识源）
        missing = sorted(result.missing_aspects)
        streak = (state.get("missing_streak", 0) + 1
                  if missing and missing == sorted(state.get("last_missing") or [])
                  else 1)
        hard_stop = (at_hop_limit
                     or streak >= VERIFY_MISSING_STREAK
                     or state.get("no_new_hits"))
        if hard_stop:
            if at_hop_limit:
                trace = trace + [_step("verify", "degraded", code="W6", message="达跳数上限仍不足")]
            trace = trace + [_step("verify", "degraded", code="W1", message="证据不足硬停",
                                   missing=list(result.missing_aspects or []))]
            return {"evidence_status": "uncertain",
                    "missing_aspects": result.missing_aspects,
                    "verify_retrying": False, "trace": trace}
        # 未写 evidence_status，必须显式置 verify_retrying 标记路由回 embed，
        # 否则二次校验时上一轮的残留值会让 route_verify 误判为已定局
        next_query = (" ".join(result.next_keywords)
                      if result.next_keywords else result.next_query)
        trace = trace + [_step("verify", "normal", message="需补充检索")]
        return {"current_query": next_query or state["rewritten_question"],
                "verify_retrying": True,
                "missing_streak": streak, "last_missing": missing, "trace": trace}

    def context_node(state: RAGState, config: RunnableConfig) -> dict:
        """精排后的Chunk → 带来源标注（含页码）的上下文文本

        按需回读（视频方法论“Agent 轻量工具组”）：仅当证据不足（uncertain
        硬停）时，用 page_tools.read_page 回读 Top 命中所在原文页，补充
        chunk 边界外的邻近内容；正常（verified）路径不回读，避免上下文膨胀。
        db 从运行时 config 注入（同 search_node，单例图不闭包捕获请求级 Session）。"""
        ranked = state["ranked"]
        context = "\n\n".join(
            f"[来源: {c['document_name']}" + (f" p{c['page']}]" if c.get("page") else "]")
            + f"\n{c['content']}"
            for c in ranked
        )
        if PAGE_READ_ENABLED and state.get("evidence_status") == "uncertain":
            db = config["configurable"].get("db")
            if db is not None:
                seen_pages, supplements = set(), []
                for c in ranked:
                    if len(seen_pages) >= PAGE_READ_MAX_PAGES:
                        break
                    doc_id, page = c.get("document_id"), c.get("page")
                    if not doc_id or not page or (doc_id, page) in seen_pages:
                        continue
                    seen_pages.add((doc_id, page))
                    pg = read_page(db, doc_id, page)
                    if pg and pg["content"].strip():
                        supplements.append(
                            f"[原文回读: {c['document_name']} p{page}]\n"
                            + pg["content"][:PAGE_READ_MAX_CHARS])
                if supplements:
                    context += "\n\n" + "\n\n".join(supplements)
        return {"context": context}

    def answer_node(state: RAGState) -> dict:
        """(上下文+问题) → prompt → model → 纯文本；
        注入长期记忆约束输出格式/风格；证据不确定（硬停）时追加对冲指令"""
        memories = format_memories(read_memories(get_memory_store()))
        extra = ""
        if state.get("evidence_status") == "uncertain":
            missing = "、".join(state.get("missing_aspects") or []) or "关键信息"
            extra = (f"\n\n注意：已检索证据不充分（缺失方面：{missing}），"
                     "检索已因客观条件终止。请用不确定语气作答，"
                     "说明证据局限与缺失的信息，不得给出确定性结论。")
        answer_chain = (
            RunnableLambda(lambda s: {"context": s["context"],
                                      "question": s["rewritten_question"],
                                      "memories": memories,
                                      "extra": extra})
            | ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "长期记忆（用户身份/偏好，回答需遵守）:\n{memories}\n\n"
                          "上下文:\n{context}\n\n问题: {question}{extra}"),
            ])
            | model
            | parser
        )
        answer = answer_chain.invoke(state)
        msg = ("证据不足硬停作答" if state.get("evidence_status") == "uncertain"
               else "正常作答")
        return {"answer": answer,
                "trace": state.get("trace", []) + [
                    _step("answer", "normal", code="N1", message=msg)]}

    # ===== 条件边：判断函数，入参为上一个节点写入后的状态 =====
    def route_permission(state: RAGState) -> str:
        return "end" if state.get("permission_denied") else "intent"

    def route_action(state: RAGState) -> str:
        # 意图先行：refuse / clarify 直接短路 END（调用方按固定话术处理），
        # 跳过改写省一次 LLM 调用；direct / search 才进改写
        return "end" if state.get("action") in ("refuse", "clarify") else "rewrite"

    def route_after_rewrite(state: RAGState) -> str:
        # 改写完成后再按 action 分流：直答 vs 检索（多跳）
        return "direct" if state.get("action") == "direct" else "search_plan"

    def route_search(state: RAGState) -> str:
        # 已判定终止 / 队列耗尽 / 达跳数上限 → 统一重排收尾；否则推理下一跳
        if (state.get("loop_finished") or not state.get("query_queue")
                or state["hop_count"] >= HOP_MAX_HOPS):
            return "rerank"
        return "hop"

    def route_after_hop(state: RAGState) -> str:
        # 推理判定终止：本跳生成的查询仍要检索（→embed），之后由
        # route_search 看到 loop_finished 收尾；必须用条件边而非固定边，
        # 且不能用队列判空（否则最后一跳的查询串会被跳过）
        return "rerank" if state.get("loop_finished") else "embed"

    def route_hits(state: RAGState) -> str:
        # 无命中直接结束（固定话术），不再触发 LLM 回答调用；有命中进证据校验
        return "verify" if state.get("ranked") else "end"

    def route_verify(state: RAGState) -> str:
        # 判定以 verify 节点本轮写入的 verify_retrying 为准（不能用
        # evidence_status 是否已写判断：同一图执行内状态会残留，
        # 二次校验时残留值会把补检循环误判为定局）
        return "embed" if state.get("verify_retrying") else "context"

    # ===== 3. 构建图 =====
    graph = StateGraph(RAGState)
    graph.add_node("permission_gate", permission_gate_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("intent", intent_node)
    graph.add_node("direct", direct_node)
    graph.add_node("search_plan", search_plan_node)
    graph.add_node("embed", embed_node)
    graph.add_node("search", search_node)
    graph.add_node("hop_reason", hop_reason_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("verify", verify_node)
    graph.add_node("context", context_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "permission_gate")
    graph.add_conditional_edges(
        "permission_gate", route_permission, {"end": END, "intent": "intent"}
    )
    graph.add_conditional_edges(
        "intent", route_action, {"end": END, "rewrite": "rewrite"}
    )
    graph.add_conditional_edges(
        "rewrite", route_after_rewrite,
        {"direct": "direct", "search_plan": "search_plan"}
    )
    graph.add_edge("direct", END)
    graph.add_edge("search_plan", "embed")
    graph.add_edge("embed", "search")
    graph.add_conditional_edges(
        "search", route_search, {"hop": "hop_reason", "rerank": "rerank"}
    )
    graph.add_conditional_edges(
        "hop_reason", route_after_hop, {"embed": "embed", "rerank": "rerank"}
    )
    graph.add_conditional_edges(
        "rerank", route_hits, {"end": END, "verify": "verify"}
    )
    graph.add_conditional_edges(
        "verify", route_verify, {"context": "context", "embed": "embed"}
    )
    graph.add_edge("context", "answer")
    graph.add_edge("answer", END)

    # 短期记忆：MemorySaver Checkpointer（状态存进程内存，程序退出自动清理，
    # 零落库）；thread_id 为前端传入的 conversation_id
    return graph.compile(checkpointer=MemorySaver())


def get_compiled_graph(api_key: str, model=None):
    """按 (api_key, id(model)) 缓存的进程级单例编译图（首次调用时构建）。
    测试注入 mock 时同 mock 复用同一图与同一 MemorySaver，
    保证多轮对话共享短期记忆。"""
    key = (api_key, id(model) if model is not None else None)
    graph = _GRAPH_CACHE.get(key)
    if graph is None:
        graph = build_rag_graph(api_key, model=model)
        _GRAPH_CACHE[key] = graph
    return graph


def _task_status(resp: QueryResponse) -> str:
    """本轮结果 → 任务状态字符串（短期记忆，供下轮 rewrite 消费）"""
    if resp.response_type == "permission_denied":
        return "权限不足，未回答"
    if resp.response_type == "refusal":
        return "已拒绝（安全风险）"
    if resp.response_type == "clarification":
        return f"等待用户澄清：{resp.clarification_question or resp.answer}"
    if resp.evidence_status == "uncertain":
        return "已回答但证据不足"
    if resp.answer == NO_RESULT_MESSAGE:
        return "知识库未找到相关信息"
    return "已完成回答"


def _remember_turn(graph, config: dict, question: str, resp: QueryResponse):
    """短期记忆写入口：所有分支终了后追加本轮问答条目与任务状态。
    as_node="answer" 表示视同终节点写入，不触发额外执行。"""
    entry = f"问：{question}\n答：{resp.answer[:_ANSWER_SNIPPET_LEN]}"
    graph.update_state(config, {
        "conversation_history": [entry],
        "last_task_status": _task_status(resp),
    }, as_node="answer")


def _extract_memories(api_key: str, model, question: str, resp: QueryResponse):
    """长期记忆写入口：所有分支终了后同步提取确定性事实写入 Store。
    记忆是增强项，任何异常静默降级，绝不阻断问答返回。"""
    try:
        store = get_memory_store()
        existing = read_memories(store)
        extractor = build_memory_extractor(api_key, model=model)
        result = extractor(question, resp.answer[:_ANSWER_SNIPPET_LEN], existing)
        apply_memory_ops(store, getattr(result, "operations", []))
    except Exception as e:
        logger.warning(f"长期记忆提取失败，跳过本轮: {e}")


def _record_trace(db, conversation_id, question, resp: QueryResponse, trace: list):
    """检索轨迹写入口：由本轮 trace 分级并落一条 QueryTrace。
    诊断是旁路增强，db 为空或写库异常一律静默跳过，绝不阻断问答返回。"""
    if db is None:
        return
    try:
        level, code = _classify_trace(trace)
        db.add(QueryTrace(
            conversation_id=conversation_id, question=question,
            answer_snippet=(resp.answer or "")[:_ANSWER_SNIPPET_LEN],
            response_type=resp.response_type, final_level=level, final_code=code,
            steps=json.dumps(trace, ensure_ascii=False),
        ))
        db.commit()
    except Exception as e:
        logger.warning(f"检索轨迹落库失败，跳过本轮: {e}")
        try:
            db.rollback()
        except Exception:
            pass


def rag_chain_query(
    db: Session, query_text: str, api_key: str, model=None, role: str = "employee",
    conversation_id: str | None = None,
) -> QueryResponse:
    """Graph 入口：执行问答状态图，并把终态映射为 QueryResponse

    conversation_id 当 Checkpointer 的 thread_id（短期记忆载体），未传兜底 "default"；
    生产/测试均复用单例编译图，请求级 db 经 RunnableConfig 注入；
    role 为员工时检索排除经理专属文档。
    """
    rag_graph = get_compiled_graph(api_key, model=model)
    config = {"configurable": {"db": db,
                               "thread_id": conversation_id or "default"}}

    # ===== 4. 执行（同 thread 时 Checkpointer 自动恢复上轮短期记忆）=====
    final = rag_graph.invoke({
        "question": query_text,
        "role": role,
    }, config=config)
    trace = final.get("trace", [])

    slots = final.get("slots")
    intent_info = _to_intent_info(slots) if slots is not None else None

    # 角色主题门禁短路：也记入短期记忆（下轮可感知上轮被拦截）
    if final.get("permission_denied"):
        resp = QueryResponse(response_type="permission_denied",
                             answer=PERMISSION_DENIED_MESSAGE)
        _remember_turn(rag_graph, config, query_text, resp)
        _extract_memories(api_key, model, query_text, resp)
        _record_trace(db, conversation_id or "default", query_text, resp, trace)
        return resp

    action = final.get("action")
    # 安全风险：固定话术拒绝，不再调用 LLM
    if action == "refuse":
        resp = QueryResponse(response_type="refusal", answer=REFUSAL_MESSAGE)
        _remember_turn(rag_graph, config, query_text, resp)
        _extract_memories(api_key, model, query_text, resp)
        _record_trace(db, conversation_id or "default", query_text, resp, trace)
        return resp

    # 信息不完整：返回澄清问题 + 部分意图分析，等待用户下轮补充；
    # 状态"等待用户澄清"供下轮 rewrite 承接补充语境
    if action == "clarify":
        question = (slots.clarification_question
                    if slots is not None and slots.clarification_question
                    else "请补充更多信息后重新提问。")
        resp = QueryResponse(
            response_type="clarification",
            answer=question,
            intent=intent_info,
            clarification_question=question,
        )
        _remember_turn(rag_graph, config, query_text, resp)
        _extract_memories(api_key, model, query_text, resp)
        _record_trace(db, conversation_id or "default", query_text, resp, trace)
        return resp

    # search 无命中：固定话术（图中已在 rerank 条件边短路，未调 LLM）
    if action == "search" and not final.get("ranked"):
        resp = QueryResponse(response_type="answer", answer=NO_RESULT_MESSAGE,
                             intent=intent_info)
        _remember_turn(rag_graph, config, query_text, resp)
        _extract_memories(api_key, model, query_text, resp)
        _record_trace(db, conversation_id or "default", query_text, resp, trace)
        return resp

    # direct / search 命中：正常回答（evidence_status 空串归一为 None）
    sources = [
        SourceInfo(
            chunk_id=c["id"],
            content_snippet=c["content"][:200],
            document_name=c["document_name"],
            page=c.get("page"),
        )
        for c in (final.get("ranked") or [])[:3]
    ]
    resp = QueryResponse(
        response_type="answer",
        answer=final.get("answer", ""),
        sources=sources,
        intent=intent_info,
        evidence_status=final.get("evidence_status") or None,
    )
    _remember_turn(rag_graph, config, query_text, resp)
    _extract_memories(api_key, model, query_text, resp)
    _record_trace(db, conversation_id or "default", query_text, resp, trace)
    return resp


def generate_answer_via_chain(
    db: Session, query_text: str, client: DeepSeekClient, role: str = "employee",
    conversation_id: str | None = None,
) -> QueryResponse:
    """兼容旧调用签名的包装函数（复用 Setting 中已存的 API Key）"""
    return rag_chain_query(db, query_text, client.api_key, role=role,
                           conversation_id=conversation_id)
