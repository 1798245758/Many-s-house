"""基于 LangChain 链式架构(LCEL 管道风格)的知识库问答 Chain

入口先过两阶段意图关卡（app/services/retrieval/intent_chain.py）：

    {"question": 问题} → intent_gate（classify_chain + slot_chain）
      ├─ refuse  → 固定礼貌拒绝话术
      ├─ clarify → 澄清问题 + 部分意图分析
      ├─ direct  → 空上下文直答
      └─ search  → 先跑检索三段（一次 invoke），无命中直接返回固定话术；
            有命中再跑生成两段，共五段：
            embed_chain      ① 关键词向量化
            search_chain     ② hybrid_search 混合检索（关键词来自槽位提取）
            rerank_chain     ③ 用 ChromaDB 已存向量算余弦相似度重排
            context_chain    ④ 拼成带来源标注的上下文
            answer_chain     ⑤ prompt | model | parser 生成回答

不依赖任何 LangGraph 组件，全部为 Runnable + | 管道组装。
"""
import math
import struct

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from sqlalchemy.orm import Session

from app.services.deepseek import DeepSeekClient
from app.services.retrieval.searcher import hybrid_search
from app.services.retrieval.intent_chain import build_intent_gate
from app.services.embedding import embed_text, EMBEDDING_DIM
from app.services import vector_store
from app.services.role import ROLE_MANAGER, manager_doc_ids
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


def build_rag_chain(db: Session, api_key: str, model=None, exclude_doc_ids: list[int] | None = None):
    """构建检索五段式 Chain（仅 search 分支使用），拆为两段返回

    返回 (retrieval_chain, answer_chain)：
    - retrieval_chain  检索三段（向量化→检索→精排），先跑完用于判断是否命中，
      避免无命中时白白消耗一次 LLM 调用、或 LLM 异常时吞掉固定兑底话术；
    - answer_chain     生成两段（上下文→回答），命中后才执行。

    exclude_doc_ids：员工角色时传入经理专属文档 ID，检索层排除。

    通过闭包捕获本次请求的 db 会话，使检索逻辑能访问数据库；
    每次请求构建新 Chain，天然适配 FastAPI 的请求级依赖注入。
    model 参数仅供测试注入 mock。
    """
    if model is None:
        model = _build_model(api_key)
    parser = StrOutputParser()

    # ① 关键词向量化 Chain：检索关键词串 → 768维查询向量
    embed_chain = RunnableLambda(lambda state: embed_query(state["search_query"]))

    # ② 检索 Chain：向量+文本混合检索（关键词来自意图关卡槽位提取，按角色排除不可见文档）
    def do_search(state: dict) -> list:
        return hybrid_search(db, state["embedding"], state["search_query"], top_k=10,
                             exclude_doc_ids=exclude_doc_ids)

    search_chain = RunnableLambda(do_search)

    # ③ 精排 Chain：从ChromaDB读候选向量，按与查询向量的余弦相似度降序重排
    def do_rerank(state: dict) -> list:
        hits = state["hits"]
        if len(hits) <= 1:
            return hits
        embeddings = vector_store.get_embeddings([c.id for c in hits])
        def score(chunk):
            vec = embeddings.get(chunk.id)
            # ChromaDB可能返回numpy数组，不能用if vec判空；缺失时记-1排最后
            return _cosine(state["embedding"], list(vec)) if vec is not None else -1.0
        return sorted(hits, key=score, reverse=True)

    rerank_chain = RunnableLambda(do_rerank)

    # ④ 上下文组织 Chain：精排后的Chunk → 带来源标注的上下文文本
    def build_context(state: dict) -> str:
        return "\n\n".join(
            f"[来源: {c.document.filename if c.document else '未知'}]\n{c.content}"
            for c in state["ranked"]
        )

    context_chain = RunnableLambda(build_context)

    # ⑤ 生成回答 Chain：(上下文+问题) → prompt → model → 纯文本
    answer_chain = (
        RunnableLambda(lambda state: {"context": state["context"], "question": state["question"]})
        | ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "上下文:\n{context}\n\n问题: {question}"),
        ])
        | model
        | parser
    )

    # 总装拆为两段：检索三段先行，命中判断后再跑生成两段
    retrieval_chain = (
        RunnablePassthrough.assign(embedding=embed_chain)
        | RunnablePassthrough.assign(hits=search_chain)
        | RunnablePassthrough.assign(ranked=rerank_chain)
    )
    generation_chain = (
        RunnablePassthrough.assign(context=context_chain)
        | RunnablePassthrough.assign(answer=answer_chain)
    )
    return retrieval_chain, generation_chain


def rag_chain_query(
    db: Session, query_text: str, api_key: str, model=None, role: str = "employee"
) -> QueryResponse:
    """Chain 入口：先过意图关卡分流，search 分支走五段式检索管道

    model 参数仅供测试注入 mock（同时传给意图关卡与回答链）；
    role 为员工时检索排除经理专属文档。
    """
    gate = build_intent_gate(api_key, model=model)
    intent = gate(query_text)
    action, slots = intent["action"], intent["slots"]

    # 安全风险：固定话术拒绝，不再调用 LLM
    if action == "refuse":
        return QueryResponse(response_type="refusal", answer=REFUSAL_MESSAGE)

    # 信息不完整：返回澄清问题 + 部分意图分析，等待用户下轮补充
    if action == "clarify":
        question = (slots.clarification_question
                    if slots and slots.clarification_question
                    else "请补充更多信息后重新提问。")
        return QueryResponse(
            response_type="clarification",
            answer=question,
            intent=_to_intent_info(slots) if slots else None,
            clarification_question=question,
        )

    # 无需检索：空上下文直答
    if action == "direct":
        direct_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "问题: {question}"),
            ])
            | (model or _build_model(api_key))
            | StrOutputParser()
        )
        answer = direct_chain.invoke({"question": query_text})
        return QueryResponse(
            response_type="answer", answer=answer, intent=_to_intent_info(slots)
        )

    # search：先跑检索三段判断命中，无命中直接固定话术（不调 LLM）
    search_query = " ".join(slots.search_keywords) if slots and slots.search_keywords else query_text
    # 员工角色：排除经理专属文档（权限收口在检索层）
    exclude_ids = None if role == ROLE_MANAGER else manager_doc_ids(db)
    retrieval_chain, generation_chain = build_rag_chain(
        db, api_key, model=model, exclude_doc_ids=exclude_ids
    )
    state = retrieval_chain.invoke({"question": query_text, "search_query": search_query})
    
    ranked = state["ranked"]
    # 需要检索但知识库完全无命中时，沿用旧链路的固定话术
    if not ranked:
        return QueryResponse(
            response_type="answer",
            answer="当前知识库中没有相关信息，请先上传文档。",
            intent=_to_intent_info(slots) if slots else None,
        )
    
    # 命中后再生成回答（上下文→回答两段）
    result = generation_chain.invoke(state)

    sources = [
        SourceInfo(
            chunk_id=c.id,
            content_snippet=c.content[:200],
            document_name=c.document.filename if c.document else "未知",
        )
        for c in ranked[:3]
    ]
    return QueryResponse(
        response_type="answer",
        answer=result["answer"],
        sources=sources,
        intent=_to_intent_info(slots) if slots else None,
    )


def generate_answer_via_chain(
    db: Session, query_text: str, client: DeepSeekClient, role: str = "employee"
) -> QueryResponse:
    """兼容旧调用签名的包装函数（复用 Setting 中已存的 API Key）"""
    return rag_chain_query(db, query_text, client.api_key, role=role)
