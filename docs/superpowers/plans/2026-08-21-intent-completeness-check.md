# 意图识别与信息完整性检查 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在在线问答链路前端增加两阶段意图关卡（分类+槽位提取），实现澄清反问、安全拒绝、直答、检索四种分流，并在前后端展示意图分析结果。

**Architecture:** 新增 `intent_chain.py`（classify_chain + slot_chain，`with_structured_output` 结构化输出，普通函数编排分流，不引入 LangGraph）；改造 `agent_chain.py` 删除原 decision_chain/bind_tools，入口先过意图关卡；QueryResponse 扩展 `response_type`/`intent`/`clarification_question` 字段；前端 Chat 页面按类型渲染。

**Tech Stack:** Python 3.14 + LangChain LCEL（init_chat_model 走 OpenAI 兼容协议接 DeepSeek）、FastAPI、Pydantic v2、React 18、pytest（unittest.mock）

**设计文档:** `docs/superpowers/specs/2026-08-21-intent-completeness-check-design.md`

**环境注意:**
- 测试从 `backend` 目录执行：`py -m pytest tests/xxx.py -v`（Windows 下 python.exe 为无效 stub，必须用 py launcher）
- pytest 需设 `USE_BGE_MODEL=false` 环境变量；conftest 已有 `isolated_vector_store` 自动隔离 ChromaDB
- PowerShell 不支持 `&&`，用 `;` 分隔
- 每个 Task 完成后提交 git（用户已同意按计划提交）

---

### Task 1: 扩展查询 Schema（IntentInfo + QueryResponse 新字段）

**Files:**
- Modify: `backend/app/schemas/query.py`（整文件仅 14 行，全量重写）
- Test: `backend/tests/test_query_schema.py`（新建）

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/test_query_schema.py`：

```python
from app.schemas.query import QueryResponse, IntentInfo


def test_intent_info_defaults():
    info = IntentInfo(task="制度咨询")
    assert info.entities == []
    assert info.time is None
    assert info.risk_note is None


def test_query_response_backward_compatible():
    r = QueryResponse(answer="ok")
    assert r.response_type == "answer"
    assert r.sources == []
    assert r.intent is None
    assert r.clarification_question is None


def test_clarification_response_fields():
    intent = IntentInfo(task="制度咨询", entities=["休假"], time="2024年")
    r = QueryResponse(
        response_type="clarification",
        answer="请补充信息",
        intent=intent,
        clarification_question="您想了解哪个部门的休假制度？",
    )
    d = r.model_dump()
    assert d["response_type"] == "clarification"
    assert d["intent"]["task"] == "制度咨询"
    assert d["intent"]["entities"] == ["休假"]
    assert d["clarification_question"] == "您想了解哪个部门的休假制度？"
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_query_schema.py -v`（在 `backend` 目录）
Expected: FAIL，`ImportError: cannot import name 'IntentInfo'`

- [ ] **Step 3: 实现 Schema**

将 `backend/app/schemas/query.py` 替换为：

```python
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class SourceInfo(BaseModel):
    chunk_id: int
    content_snippet: str
    document_name: str


class IntentInfo(BaseModel):
    """意图分析结果（任务/实体/时间/风险）"""
    task: str
    entities: list[str] = []
    time: str | None = None
    risk_note: str | None = None


class QueryResponse(BaseModel):
    response_type: str = "answer"            # answer | clarification | refusal
    answer: str                              # 回答 / 澄清引导语 / 拒绝说明
    sources: list[SourceInfo] = []
    intent: IntentInfo | None = None
    clarification_question: str | None = None
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_query_schema.py -v`
Expected: 3 passed

- [ ] **Step 5: 提交**

```bash
git add backend/app/schemas/query.py backend/tests/test_query_schema.py
git commit -m "feat: QueryResponse 增加 intent/response_type/clarification_question 字段"
```

---

### Task 2: 意图链模块（classify_chain + slot_chain + intent_gate）

**Files:**
- Create: `backend/app/services/retrieval/intent_chain.py`
- Test: `backend/tests/test_intent_chain.py`（新建）

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/test_intent_chain.py`（mock LLM 用 `unittest.mock.patch` 替换 `init_chat_model`；`with_structured_output` 返回 MagicMock，其 `invoke` 返回指定对象）：

```python
from unittest.mock import MagicMock, patch

from app.services.retrieval.intent_chain import build_intent_gate


def _mock_model(classify_ret, slot_ret=None):
    """让 with_structured_output 按调用顺序返回 classify/slot 两个链 mock

    注意：实现中链为 prompt | structured_output 管道，invoke 时管道会
    把 mock 当函数调用，所以用 return_value 而非 invoke.return_value。
    """
    model = MagicMock()
    cls_mock = MagicMock()
    cls_mock.return_value = classify_ret
    slot_mock = MagicMock()
    slot_mock.return_value = slot_ret
    model.with_structured_output.side_effect = [cls_mock, slot_mock]
    return model


def test_refuse_action():
    from app.services.retrieval.intent_chain import ClassifyResult
    model = _mock_model(ClassifyResult(action="refuse", risk_reason="涉及违法"))
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("违法问题")
    assert out["action"] == "refuse"
    assert out["risk_reason"] == "涉及违法"
    assert out["slots"] is None


def test_search_action_extracts_keywords():
    from app.services.retrieval.intent_chain import ClassifyResult, SlotResult
    slots = SlotResult(task="制度咨询", entities=["休假"], time="", risk_note="",
                       search_keywords=["员工", "休假", "制度"], clarification_question="")
    model = _mock_model(ClassifyResult(action="search", risk_reason=""), slots)
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("胖东来的员工休假制度")
    assert out["action"] == "search"
    assert out["slots"].search_keywords == ["员工", "休假", "制度"]


def test_classify_failure_degrades_to_search():
    model = MagicMock()
    model.with_structured_output.return_value.side_effect = RuntimeError("LLM 挂了")
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("任意问题")
    assert out["action"] == "search"
    assert out["slots"].search_keywords == ["任意问题"]


def test_slot_failure_falls_back_to_question():
    from app.services.retrieval.intent_chain import ClassifyResult
    cls_mock = MagicMock()
    cls_mock.return_value = ClassifyResult(action="search", risk_reason="")
    slot_mock = MagicMock()
    slot_mock.side_effect = RuntimeError("解析失败")
    model = MagicMock()
    # with_structured_output(schema) 依次返回 classify/slot 两个 runnable
    model.with_structured_output.side_effect = [cls_mock, slot_mock]
    gate = build_intent_gate(api_key="sk-test", model=model)
    out = gate("任意问题")
    assert out["action"] == "search"
    assert out["slots"].search_keywords == ["任意问题"]
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_intent_chain.py -v`
Expected: FAIL，`ModuleNotFoundError: app.services.retrieval.intent_chain`

- [ ] **Step 3: 实现意图链**

创建 `backend/app/services/retrieval/intent_chain.py`：

```python
"""意图识别与信息完整性检查 Chain（两阶段 LLM 调用）

① classify_chain  分类：判断问题动作 refuse/clarify/direct/search + 风险理由
② slot_chain      槽位提取：任务/实体/时间/风险备注 + 检索关键词或澄清问题

普通函数 build_intent_gate 编排两阶段并按 action 分流，不依赖 LangGraph。
任一阶段失败均降级为"以原问题作关键词走检索"，保证可用性优先。
"""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = (
    "你是企业知识库助手的前置分析器。分析用户问题，选择唯一动作：\n"
    "refuse: 问题涉及违法、暴力、色情、涉密等有害或敏感内容；\n"
    "clarify: 问题意图模糊到无法判断用户在问什么，或缺失的关键信息会直接导致"
    "歧义、无法检索（采用宽松标准，一般性问题不要选此项）；\n"
    "direct: 闲聊、问候、常识类问题，无需企业知识库资料即可回答；\n"
    "search: 涉及企业管理事实（文化、制度、培训、薪酬福利、运营流程等），需要检索。\n"
    "若动作是 refuse，在 risk_reason 中给出简要理由；否则 risk_reason 留空字符串。\n"
    "问题: {question}"
)

SLOT_PROMPT = (
    "从用户问题中提取信息。动作类型: {action}\n"
    "task: 任务类型（如 制度咨询、流程咨询、闲聊，简短词语）；\n"
    "entities: 关键实体列表（机构、制度名、人员角色等），没有则空列表；\n"
    "time: 时间信息，没有则留空字符串；\n"
    "risk_note: 问题涉及的潜在风险备注，没有则留空字符串；\n"
    "search_keywords: 仅当动作为 search 时填写，由问题改写的2~6个核心检索关键词，"
    "否则留空列表；\n"
    "clarification_question: 仅当动作为 clarify 时填写，一句面向用户的澄清问题"
    "（指出缺失的信息并请用户补充），否则留空字符串。\n"
    "问题: {question}"
)


class ClassifyResult(BaseModel):
    action: str = Field(description="refuse|clarify|direct|search")
    risk_reason: str = ""


class SlotResult(BaseModel):
    task: str = ""
    entities: list[str] = []
    time: str = ""
    risk_note: str = ""
    search_keywords: list[str] = []
    clarification_question: str = ""


def build_intent_gate(api_key: str, model=None):
    """构建意图关卡：返回 gate(question) -> dict

    返回 dict 结构：{"action", "risk_reason", "slots"(SlotResult|None), "question"}
    model 参数仅供测试注入 mock；缺省时初始化 DeepSeek（OpenAI 兼容协议）。
    """
    if model is None:
        model = init_chat_model(
            "deepseek-chat",
            model_provider="openai",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0.3,
        )

    classify_chain = (
        ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
        | model.with_structured_output(ClassifyResult)
    )
    slot_chain = (
        ChatPromptTemplate.from_template(SLOT_PROMPT)
        | model.with_structured_output(SlotResult)
    )

    def fallback_slots(question: str) -> SlotResult:
        return SlotResult(task="未知", search_keywords=[question])

    def intent_gate(question: str) -> dict:
        # 第一阶段：分类；失败降级为 search + 原问题关键词
        try:
            cls = classify_chain.invoke({"question": question})
        except Exception as e:
            logger.warning(f"意图分类失败，降级为检索: {e}")
            return {"action": "search", "risk_reason": "",
                    "slots": fallback_slots(question), "question": question}

        if cls.action == "refuse":
            return {"action": "refuse", "risk_reason": cls.risk_reason,
                    "slots": None, "question": question}

        # 第二阶段：槽位提取；失败用原问题兜底关键词
        try:
            slots = slot_chain.invoke({"question": question, "action": cls.action})
        except Exception as e:
            logger.warning(f"槽位提取失败，用原问题兜底: {e}")
            slots = fallback_slots(question)

        return {"action": cls.action, "risk_reason": cls.risk_reason,
                "slots": slots, "question": question}

    return intent_gate
```

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_intent_chain.py -v`
Expected: 4 passed

- [ ] **Step 5: 提交**

```bash
git add backend/app/services/retrieval/intent_chain.py backend/tests/test_intent_chain.py
git commit -m "feat: 新增两阶段意图链 intent_chain（分类+槽位提取+降级）"
```

---

### Task 3: 改造 agent_chain.py（接入意图关卡，删除 decision_chain）

**Files:**
- Modify: `backend/app/services/retrieval/agent_chain.py`
- Test: `backend/tests/test_agent_chain.py`（新建）

- [ ] **Step 1: 写失败测试**

创建 `backend/tests/test_agent_chain.py`（注入 mock 模型，验证四种分流；`USE_BGE_MODEL=false` 下 embed 为确定性假向量，空库检索命中为空）：

```python
from unittest.mock import MagicMock, patch

from app.services.retrieval.agent_chain import build_rag_chain, rag_chain_query
from app.services.retrieval.intent_chain import ClassifyResult, SlotResult


def _mock_model(action: str, **slot_kwargs):
    """mock：with_structured_output 依次返回 classify/slot 两个链；
    model.invoke 供 answer_chain/direct_chain 管道末端调用"""
    cls_mock = MagicMock()
    cls_mock.return_value = ClassifyResult(action=action, risk_reason="测试风险")

    slot_mock = MagicMock()
    slot_mock.return_value = SlotResult(**slot_kwargs)

    model = MagicMock()
    # 意图链：prompt | structured_output 管道把 mock 当函数调用
    model.with_structured_output.side_effect = [cls_mock, slot_mock]
    # 回答链：prompt | model | parser 管道同样把 model 当函数调用
    model.return_value = "直答内容"
    return model


def test_refuse_returns_refusal():
    model = _mock_model("refuse")
    resp = rag_chain_query(None, "有害问题", "sk-test", model=model)
    assert resp.response_type == "refusal"
    assert resp.intent is None


def test_clarify_returns_clarification():
    model = _mock_model("clarify", task="制度咨询", entities=["休假"],
                        clarification_question="您想了解哪个部门？")
    resp = rag_chain_query(None, "那个制度", "sk-test", model=model)
    assert resp.response_type == "clarification"
    assert resp.clarification_question == "您想了解哪个部门？"
    assert resp.intent.task == "制度咨询"


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
```

- [ ] **Step 2: 运行确认失败**

Run: `py -m pytest tests/test_agent_chain.py -v`
Expected: FAIL（`rag_chain_query` 不接受 `model` 参数 / 响应无 `response_type`）

- [ ] **Step 3: 重写 agent_chain.py**

将 `backend/app/services/retrieval/agent_chain.py` 全量替换为：

```python
"""基于 LangChain 链式架构(LCEL 管道风格)的知识库问答 Chain

入口先过两阶段意图关卡（app/services/retrieval/intent_chain.py）：

    {"question": 问题} → intent_gate（classify_chain + slot_chain）
      ├─ refuse  → 固定礼貌拒绝话术
      ├─ clarify → 澄清问题 + 部分意图分析
      ├─ direct  → answer_chain 空上下文直答
      └─ search  → 五段式管道（一次 invoke）：
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


def embed_query(text: str) -> list[float]:
    """查询文本 → BGE 向量"""
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


def build_rag_chain(db: Session, api_key: str, model=None):
    """构建检索五段式 Chain（仅 search 分支使用）

    通过闭包捕获本次请求的 db 会话；每次请求构建新 Chain，
    天然适配 FastAPI 请求级依赖注入。model 参数仅供测试注入 mock。
    """
    if model is None:
        model = init_chat_model(
            "deepseek-chat",
            model_provider="openai",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0.7,
        )
    parser = StrOutputParser()

    # ① 关键词向量化 Chain：检索关键词串 → 768维查询向量
    embed_chain = RunnableLambda(lambda state: embed_query(state["search_query"]))

    # ② 检索 Chain：向量+文本混合检索（关键词来自意图关卡槽位提取）
    def do_search(state: dict) -> list:
        return hybrid_search(db, state["embedding"], state["search_query"], top_k=10)

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

    # 总装：assign 把每步产出挂进状态字典，逐级累积，一次 invoke 跑完
    rag_chain = (
        RunnablePassthrough.assign(embedding=embed_chain)
        | RunnablePassthrough.assign(hits=search_chain)
        | RunnablePassthrough.assign(ranked=rerank_chain)
        | RunnablePassthrough.assign(context=context_chain)
        | RunnablePassthrough.assign(answer=answer_chain)
    )
    return rag_chain


def rag_chain_query(
    db: Session, query_text: str, api_key: str, model=None
) -> QueryResponse:
    """Chain 入口：先过意图关卡分流，search 分支走五段式检索管道

    model 参数仅供测试注入 mock（同时传给意图关卡与回答链）。
    """
    gate = build_intent_gate(api_key, model=model)
    intent = gate(query_text)
    action, slots = intent["action"], intent["slots"]

    if action == "refuse":
        return QueryResponse(response_type="refusal", answer=REFUSAL_MESSAGE)

    if action == "clarify":
        question = slots.clarification_question or "请补充更多信息后重新提问。"
        return QueryResponse(
            response_type="clarification",
            answer=question,
            intent=_to_intent_info(slots),
            clarification_question=question,
        )

    if action == "direct":
        direct_chain = (
            ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "问题: {question}"),
            ])
            | (model or init_chat_model(
                "deepseek-chat", model_provider="openai",
                base_url="https://api.deepseek.com",
                api_key=api_key, temperature=0.7))
            | StrOutputParser()
        )
        answer = direct_chain.invoke({"question": query_text})
        return QueryResponse(
            response_type="answer", answer=answer, intent=_to_intent_info(slots)
        )

    # action == "search"：五段式管道
    search_query = " ".join(slots.search_keywords) if slots.search_keywords else query_text
    chain = build_rag_chain(db, api_key, model=model)
    result = chain.invoke({"question": query_text, "search_query": search_query})

    ranked = result["ranked"]
    if not ranked:
        return QueryResponse(
            response_type="answer",
            answer="当前知识库中没有相关信息，请先上传文档。",
            intent=_to_intent_info(slots),
        )

    sources = [
        SourceInfo(
            chunk_id=c.id,
            content_snippet=c.content[:200],
            document_name=c.document.filename if c.document else "未知",
        )
        for c in ranked[:3]
    ]
    return QueryResponse(
        response_type="answer", answer=result["answer"],
        sources=sources, intent=_to_intent_info(slots),
    )


def generate_answer_via_chain(
    db: Session, query_text: str, client: DeepSeekClient
) -> QueryResponse:
    """兼容旧调用签名的包装函数（复用 Setting 中已存的 API Key）"""
    return rag_chain_query(db, query_text, client.api_key)
```

要点核对：
- 已删除 `DECISION_PROMPT`、`decision_chain`、`bind_tools`、`@tool`、`exec_tool`
- 检索关键词来自 `slots.search_keywords`，精排向量基准同为该关键词串向量（与原 exec_tool 行为一致）
- `query.py` 路由无需改动（`generate_answer_via_chain` 签名未变）

- [ ] **Step 4: 运行确认通过**

Run: `py -m pytest tests/test_agent_chain.py -v`
Expected: 4 passed

- [ ] **Step 5: 回归全量测试**

Run: `py -m pytest tests/ -v`
Expected: 全部 PASS（确认删除 decision_chain 未破坏其他测试）

- [ ] **Step 6: 提交**

```bash
git add backend/app/services/retrieval/agent_chain.py backend/tests/test_agent_chain.py
git commit -m "feat: 问答入口接入意图关卡，decision_chain 替换为两阶段意图链"
```

---

### Task 4: 前端 Chat 页面改造（意图标签 + 澄清/拒绝样式）

**Files:**
- Modify: `frontend/src/pages/Chat/index.jsx`

`api/query.js` 与 `api/index.js` 已透传 `data.data`，无需改动。

- [ ] **Step 1: 更新状态与请求处理**

将 `frontend/src/pages/Chat/index.jsx` 全量替换为：

```jsx
import { useState, useEffect } from 'react'
import { useSearchParams } from 'react-router-dom'
import { submitQuery } from '../../api/query'
import Loading from '../../components/Loading'
import ErrorMessage from '../../components/ErrorMessage'

const tagBase = {display:'inline-block',padding:'2px 8px',borderRadius:12,fontSize:'0.8rem',marginRight:6,marginBottom:4}

function IntentTags({ intent }) {
  if (!intent) return null
  return (
    <div style={{marginBottom:10}}>
      <small style={{color:'#666',marginRight:8}}>意图分析:</small>
      <span style={{...tagBase,background:'#e6f0ff',color:'#1a56db'}}>任务: {intent.task}</span>
      {intent.entities.map((e, i) => (
        <span key={i} style={{...tagBase,background:'#f0f0f0',color:'#333'}}>{e}</span>
      ))}
      {intent.time && <span style={{...tagBase,background:'#e6fff0',color:'#0a7a4b'}}>时间: {intent.time}</span>}
      {intent.risk_note && <span style={{...tagBase,background:'#fff3e0',color:'#b45309'}}>风险: {intent.risk_note}</span>}
    </div>
  )
}

export default function Chat() {
  const [searchParams] = useSearchParams()
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    const q = searchParams.get('q')
    if (q) {
      setQuestion(q)
      handleAsk(q)
    }
  }, [])

  const handleAsk = async (text) => {
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const data = await submitQuery(text)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!question.trim()) return
    handleAsk(question)
  }

  const type = result?.response_type || 'answer'

  return (
    <div>
      <h1 className="page-title">提问</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          rows={3}
          placeholder="输入你的问题，例如：胖东来的员工休假制度是怎样的？"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          style={{marginBottom:12}}
        />
        <button className="btn btn-primary" type="submit" disabled={loading}>
          {loading ? '查询中...' : '发送'}
        </button>
      </form>
      {error && <ErrorMessage message={error} onRetry={() => handleSubmit({ preventDefault: () => {} })} />}
      {loading && <Loading />}
      {result && type === 'clarification' && (
        <div className="card" style={{marginTop:20,border:'1px solid #f59e0b',background:'#fffbeb'}}>
          <IntentTags intent={result.intent} />
          <h3>需要补充信息</h3>
          <div className="answer-text">{result.clarification_question || result.answer}</div>
          <button
            className="btn"
            style={{marginTop:10}}
            onClick={() => setQuestion(result.clarification_question || '')}
          >
            补充后重新提问
          </button>
        </div>
      )}
      {result && type === 'refusal' && (
        <div className="card" style={{marginTop:20,border:'1px solid #ef4444',background:'#fef2f2'}}>
          <h3>无法回答</h3>
          <div className="answer-text" style={{color:'#b91c1c'}}>{result.answer}</div>
        </div>
      )}
      {result && type === 'answer' && (
        <div className="card" style={{marginTop:20}}>
          <IntentTags intent={result.intent} />
          <h3>回答</h3>
          <div className="answer-text">{result.answer}</div>
          {(result.sources || []).length > 0 && (
            <details>
              <summary>参考来源 ({result.sources.length})</summary>
              {result.sources.map((s, i) => (
                <div key={i} style={{marginTop:8,padding:8,background:'#f5f5f5',borderRadius:4}}>
                  <small style={{color:'#666'}}>{s.document_name}</small>
                  <p style={{fontSize:'0.9rem'}}>{s.content_snippet}</p>
                </div>
              ))}
            </details>
          )}
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 2: 构建验证无语法错误**

Run: `npm run build`（在 `frontend` 目录）
Expected: 构建成功，无报错

- [ ] **Step 3: 提交**

```bash
git add frontend/src/pages/Chat/index.jsx
git commit -m "feat: 问答页展示意图分析标签与澄清/拒绝样式"
```

---

### Task 5: 端到端手动验证

- [ ] **Step 1: 启动服务**

Run: `npm run dev`（在 `frontend` 目录，一键同启后端 8000 + 前端 5173）

- [ ] **Step 2: 浏览器验证四类分流**（需后端已配置有效 DeepSeek API Key）

| 输入 | 预期 response_type | 预期表现 |
|---|---|---|
| `你好` | answer | 直答，意图标签显示任务=闲聊 |
| `胖东来的员工休假制度是怎样的？` | answer | 检索回答+参考来源，标签含实体 |
| `说说那个制度` | clarification | 琥珀色卡片+澄清问题，"补充后重新提问"按钮回填输入框 |
| 一条明显违法/有害的请求 | refusal | 红色卡片拒绝说明 |

- [ ] **Step 3: 验证降级路径**

临时将个人中心 API Key 改为无效值并提问 → 预期返回 `API_KEY_MISSING`（网关层拦截，属正常）；意图链异常的降级路径已由单元测试覆盖。

- [ ] **Step 4: 最终回归**

Run: `py -m pytest tests/ -v`（`backend` 目录，`USE_BGE_MODEL=false`）
Expected: 全部 PASS

---

## Self-Review 记录

1. **Spec 覆盖**：①无状态澄清 ✅（Task 1/3/4）②安全拒绝固定话术 ✅（Task 3 REFUSAL_MESSAGE）③替代 decision_chain ✅（Task 3 删除 bind_tools/@tool）④前后端展示 ✅（Task 1 intent 字段 + Task 4 IntentTags）⑤宽松标准 ✅（Task 2 CLASSIFY_PROMPT）⑥两阶段调用与降级 ✅（Task 2）⑦历史记录落库 ✅（路由不变，answer_text 自动存澄清/拒绝文本）
2. **占位符扫描**：无 TBD/TODO，所有代码步骤含完整代码
3. **类型一致性**：`ClassifyResult`/`SlotResult`/`IntentInfo` 字段在 Task 2/3 中定义与使用一致；`rag_chain_query(db, query_text, api_key, model=None)` 签名在测试与实现一致
