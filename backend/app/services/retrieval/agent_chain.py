"""基于 LangGraph 状态图的知识库问答 Graph

===== 图结构 =====
START → permission_gate（确定性门禁，不调 LLM）
          ├─ 员工命中经理专属主题 → END（permission_denied）
          └─ 放行 → intent（意图关卡子图：classify + slot）
intent 条件边按 action 分流：
          ├─ refuse  → END（固定拒绝话术）
          ├─ clarify → END（澄清问题 + 意图分析）
          ├─ direct  → direct 节点（空上下文直答）→ END
          └─ search  → embed → search → rerank
                        ├─ 无命中 → END（固定话术，不调 LLM）
                        └─ 有命中 → context → answer → END

检索与生成分离在条件边上：rerank 后若无命中直接结束，
避免无命中时白白消耗一次 LLM 调用、或 LLM 异常时吞掉固定兜底话术。
不依赖 LangGraph 的 checkpointer（单轮无状态），全部为节点 + 条件边组装。
"""
import math
import struct
from typing import TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, START, END
from sqlalchemy.orm import Session

from app.services.deepseek import DeepSeekClient
from app.services.retrieval.searcher import hybrid_search
from app.services.retrieval.intent_chain import build_intent_gate
from app.services.embedding import embed_text, EMBEDDING_DIM
from app.services import vector_store
from app.services.role import ROLE_MANAGER, manager_doc_ids
from app.config import MANAGER_KEYWORDS
from app.schemas.query import SourceInfo, QueryResponse, IntentInfo

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
    question: str                # 用户原始问题
    role: str                    # employee | manager
    permission_denied: bool      # 角色主题门禁命中
    action: str                  # refuse | clarify | direct | search
    slots: object                # SlotResult | None（意图关卡槽位）
    search_query: str            # 检索关键词串（来自槽位或原问题兜底）
    embedding: list[float]       # 查询向量
    hits: list                   # hybrid_search 原始命中
    ranked: list                 # 余弦相似度重排后的 Chunk 列表
    context: str                 # 带来源标注的上下文
    answer: str                  # 最终回答文本


def build_rag_graph(db: Session, api_key: str, model=None):
    """构建问答状态图并编译返回

    通过闭包捕获本次请求的 db 会话与 model，每次请求构建新图，
    天然适配 FastAPI 的请求级依赖注入。model 参数仅供测试注入 mock。
    """
    if model is None:
        model = _build_model(api_key)
    parser = StrOutputParser()

    # ===== 2. 定义节点：每个节点是一个函数，读 State、写 State =====
    def permission_gate_node(state: RAGState) -> dict:
        """角色主题门禁：员工提问命中经理专属关键词直接短路，不调 LLM"""
        if state["role"] != ROLE_MANAGER and any(
            kw in state["question"] for kw in MANAGER_KEYWORDS
        ):
            return {"permission_denied": True, "answer": PERMISSION_DENIED_MESSAGE}
        return {"permission_denied": False}

    def intent_node(state: RAGState) -> dict:
        """意图关卡（classify + slot 子图），并改写检索关键词

        门禁短路路径不会到达本节点，故在此惰性构建关卡，
        避免被拦截的请求白白初始化结构化输出链。
        """
        gate = build_intent_gate(api_key, model=model)
        intent = gate(state["question"])
        slots = intent["slots"]
        search_query = (
            " ".join(slots.search_keywords)
            if slots is not None and slots.search_keywords
            else state["question"]
        )
        return {"action": intent["action"], "slots": slots, "search_query": search_query}

    def direct_node(state: RAGState) -> dict:
        """direct 分支：空上下文直答"""
        direct_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "问题: {question}"),
            ])
            | model
            | parser
        )
        return {"answer": direct_chain.invoke({"question": state["question"]})}

    def embed_node(state: RAGState) -> dict:
        """检索关键词串 → 768维查询向量"""
        return {"embedding": embed_query(state["search_query"])}

    def search_node(state: RAGState) -> dict:
        """向量+文本混合检索（员工角色排除经理专属文档）"""
        exclude = manager_doc_ids(db) if state["role"] != ROLE_MANAGER else None
        hits = hybrid_search(db, state["embedding"], state["search_query"], top_k=10,
                             exclude_doc_ids=exclude)
        return {"hits": hits}

    def rerank_node(state: RAGState) -> dict:
        """从ChromaDB读候选向量，按与查询向量的余弦相似度降序重排"""
        hits = state["hits"]
        if len(hits) <= 1:
            return {"ranked": hits}
        embeddings = vector_store.get_embeddings([c.id for c in hits])

        def score(chunk):
            vec = embeddings.get(chunk.id)
            # ChromaDB可能返回numpy数组，不能用if vec判空；缺失时记-1排最后
            return _cosine(state["embedding"], list(vec)) if vec is not None else -1.0

        return {"ranked": sorted(hits, key=score, reverse=True)}

    def context_node(state: RAGState) -> dict:
        """精排后的Chunk → 带来源标注的上下文文本"""
        context = "\n\n".join(
            f"[来源: {c.document.filename if c.document else '未知'}]\n{c.content}"
            for c in state["ranked"]
        )
        return {"context": context}

    def answer_node(state: RAGState) -> dict:
        """(上下文+问题) → prompt → model → 纯文本"""
        answer_chain = (
            RunnableLambda(lambda s: {"context": s["context"], "question": s["question"]})
            | ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "上下文:\n{context}\n\n问题: {question}"),
            ])
            | model
            | parser
        )
        return {"answer": answer_chain.invoke(state)}

    # ===== 条件边：判断函数，入参为上一个节点写入后的状态 =====
    def route_permission(state: RAGState) -> str:
        return "end" if state.get("permission_denied") else "intent"

    def route_action(state: RAGState) -> str:
        action = state.get("action")
        if action == "search":
            return "embed"
        if action == "direct":
            return "direct"
        return "end"  # refuse / clarify 由调用方按固定话术处理

    def route_hits(state: RAGState) -> str:
        # 无命中直接结束（固定话术），不再触发 LLM 回答调用
        return "context" if state.get("ranked") else "end"

    # ===== 3. 构建图 =====
    graph = StateGraph(RAGState)
    graph.add_node("permission_gate", permission_gate_node)
    graph.add_node("intent", intent_node)
    graph.add_node("direct", direct_node)
    graph.add_node("embed", embed_node)
    graph.add_node("search", search_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("context", context_node)
    graph.add_node("answer", answer_node)

    graph.add_edge(START, "permission_gate")
    graph.add_conditional_edges(
        "permission_gate", route_permission, {"end": END, "intent": "intent"}
    )
    graph.add_conditional_edges(
        "intent", route_action, {"end": END, "direct": "direct", "embed": "embed"}
    )
    graph.add_edge("direct", END)
    graph.add_edge("embed", "search")
    graph.add_edge("search", "rerank")
    graph.add_conditional_edges(
        "rerank", route_hits, {"end": END, "context": "context"}
    )
    graph.add_edge("context", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


def rag_chain_query(
    db: Session, query_text: str, api_key: str, model=None, role: str = "employee"
) -> QueryResponse:
    """Graph 入口：执行问答状态图，并把终态映射为 QueryResponse

    model 参数仅供测试注入 mock（同时传给意图关卡与回答链）；
    role 为员工时检索排除经理专属文档。
    """
    rag_graph = build_rag_graph(db, api_key, model=model)

    # ===== 4. 执行 =====
    final = rag_graph.invoke({
        "question": query_text,
        "role": role,
    })

    slots = final.get("slots")
    intent_info = _to_intent_info(slots) if slots is not None else None

    # 角色主题门禁短路
    if final.get("permission_denied"):
        return QueryResponse(response_type="permission_denied",
                             answer=PERMISSION_DENIED_MESSAGE)

    action = final.get("action")
    # 安全风险：固定话术拒绝，不再调用 LLM
    if action == "refuse":
        return QueryResponse(response_type="refusal", answer=REFUSAL_MESSAGE)

    # 信息不完整：返回澄清问题 + 部分意图分析，等待用户下轮补充
    if action == "clarify":
        question = (slots.clarification_question
                    if slots is not None and slots.clarification_question
                    else "请补充更多信息后重新提问。")
        return QueryResponse(
            response_type="clarification",
            answer=question,
            intent=intent_info,
            clarification_question=question,
        )

    # search 无命中：固定话术（图中已在 rerank 条件边短路，未调 LLM）
    if action == "search" and not final.get("ranked"):
        return QueryResponse(response_type="answer", answer=NO_RESULT_MESSAGE,
                             intent=intent_info)

    # direct / search 命中：正常回答
    sources = [
        SourceInfo(
            chunk_id=c.id,
            content_snippet=c.content[:200],
            document_name=c.document.filename if c.document else "未知",
        )
        for c in (final.get("ranked") or [])[:3]
    ]
    return QueryResponse(
        response_type="answer",
        answer=final.get("answer", ""),
        sources=sources,
        intent=intent_info,
    )


def generate_answer_via_chain(
    db: Session, query_text: str, client: DeepSeekClient, role: str = "employee"
) -> QueryResponse:
    """兼容旧调用签名的包装函数（复用 Setting 中已存的 API Key）"""
    return rag_chain_query(db, query_text, client.api_key, role=role)
