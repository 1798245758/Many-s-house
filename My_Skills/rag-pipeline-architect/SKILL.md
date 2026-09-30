---
name: rag-pipeline-architect
description: Design and implement production-grade RAG (Retrieval-Augmented Generation) systems using LangGraph state graphs. Covers six-stage pipeline architecture, intent gateway with two-stage classification, multi-hop retrieval with reasoning loops, evidence verification with hard-stop conditions, hybrid search (vector + keyword), cosine reranking, query rewrite with short-term memory, and graceful degradation strategies. Use when building RAG pipelines, designing retrieval-augmented Q&A systems, implementing multi-hop reasoning, or architecting knowledge base assistants with LangChain/LangGraph.
---

# RAG Pipeline Architect

## When to Use

- Building or refactoring a RAG retrieval-augmented generation system
- Designing multi-stage query pipelines with LangGraph state graphs
- Implementing intent classification gateways for Q&A routing
- Adding multi-hop retrieval with evidence verification loops
- Architecting hybrid search (vector + keyword) with reranking
- Designing graceful degradation for LLM-dependent pipelines

## Required Tools & Dependencies

| Category | Required | Version Constraint |
|----------|----------|--------------------|
| Orchestration | `langgraph` | >=0.2 (StateGraph + conditional_edges) |
| Chain primitives | `langchain-core` | >=0.3 (LCEL, ChatPromptTemplate, StrOutputParser) |
| Model init | `langchain` | init_chat_model with model_provider="openai" |
| Structured output | `pydantic` | v2 (BaseModel + Field) |
| Vector store | `chromadb` | >=1.5 PersistentClient (embedded, no server) |
| Embedding | `sentence-transformers` | BAAI/bge-base-zh-v1.5 (768d) or equivalent |
| Keyword search | SQLite FTS5 | Built into Python sqlite3 |
| ORM | `sqlalchemy` | >=2.0 (Session, declarative models) |
| API framework | `fastapi` | >=0.100 |

## Tool Usage Constraints (Anti-Misuse)

### MUST use:
- `model.with_structured_output(Schema, method="function_calling")` — for ALL structured LLM calls
- `StateGraph` + `add_conditional_edges` — for any flow with branching or cycles
- `RunnableConfig["configurable"]` — for injecting request-scoped resources (DB session)
- `MemorySaver` checkpointer — for short-term conversation memory
- Pure dict state fields — all state must be msgpack-serializable

### MUST NOT use:
- ❌ `method="json_schema"` — DeepSeek and most OpenAI-compatible APIs don't support it
- ❌ `json_mode` / `response_format={"type": "json_object"}` — unreliable for complex schemas
- ❌ ORM objects in state — SQLAlchemy models can't serialize, cause checkpoint crashes
- ❌ Closure-captured DB sessions — singleton graph + closure = session leak across requests
- ❌ `langchain.agents` (AgentExecutor) — non-deterministic, use explicit StateGraph instead
- ❌ Recursive LCEL chains for loops — use StateGraph cycles, they have built-in termination
- ❌ `vector` as LLM tool parameter — LLM can't generate 768-dim float arrays; embed inside the tool

### Common Agent Mistakes:

| Mistake | Why Wrong | Correct Approach |
|---------|-----------|------------------|
| Passing embedding vector to LLM tool | LLM outputs text, not float arrays | Tool takes `query: str`, embeds internally |
| Using AgentExecutor for RAG | Non-deterministic routing, no cycle control | Explicit StateGraph with conditional edges |
| Storing full ORM Chunk in state | msgpack serialization fails on lazy-loaded relations | Convert to `{id, content, document_name}` dict immediately |
| Building graph per-request | 10-50ms compilation overhead × every call | Singleton `_GRAPH_CACHE` + config injection |
| Skipping state reset on new turn | Checkpointer restores old terminal state → loop skipped | `search_plan_node` resets ALL loop fields |
| Using `evidence_status` for routing | Stale value from previous verify iteration | Use explicit `verify_retrying` flag per iteration |
| `if vec:` to check numpy array | numpy arrays raise ValueError on truthiness | Use `if vec is not None:` |

## Core Architecture: LangGraph State Graph

The RAG pipeline is modeled as a **LangGraph StateGraph** with typed state flowing between nodes. Each node reads state, performs one responsibility, and writes partial state updates.

### Graph Topology

```
START → permission_gate → intent → [route_action]
                                     ├─ refuse  → END
                                     ├─ clarify → END
                                     └─ direct/search → rewrite → [route_after_rewrite]
                                                                    ├─ direct → direct_node → END
                                                                    └─ search → search_plan → embed → search
                                                                                              ↓
                                                                                    [route_search] ←── hop_reason
                                                                                       ├─ hop → hop_reason → [route_after_hop]
                                                                                       │                         ├─ embed (continue)
                                                                                       │                         └─ rerank (finished)
                                                                                       └─ rerank → [route_hits]
                                                                                                     ├─ no hits → END
                                                                                                     └─ verify → [route_verify]
                                                                                                                   ├─ embed (retry)
                                                                                                                   └─ context → answer → END
```

### Design Principles

1. **State accumulation**: Each node returns a partial dict merged into shared state
2. **Conditional edges over hardcoded flows**: Routing logic in pure functions, testable independently
3. **Fail-safe degradation**: Every LLM call wrapped in try/except with deterministic fallback
4. **Singleton compiled graph**: Build once, inject request-scoped deps via `RunnableConfig`
5. **Pure-data state**: All state fields must be msgpack-serializable (no ORM objects)

## State Definition Pattern

```python
from typing import Annotated, TypedDict

class RAGState(TypedDict, total=False):
    # Input
    question: str
    role: str
    # Intent
    action: str              # refuse | clarify | direct | search
    slots: object            # Structured slot extraction result
    # Rewrite
    rewritten_question: str
    sub_questions: list
    rewrite_keywords: list
    search_query: str        # Merged keywords for first hop
    # Multi-hop loop
    current_query: str
    query_queue: list
    hop_count: int
    all_hits: list           # Accumulated candidates (pure dicts)
    loop_finished: bool
    no_new_hits: bool
    # Verification
    verify_retrying: bool
    evidence_status: str     # verified | uncertain
    missing_aspects: list
    missing_streak: int
    last_missing: list
    # Output
    embedding: list[float]
    ranked: list
    context: str
    answer: str
    # Short-term memory
    conversation_history: Annotated[list, _keep_recent_history]
    last_task_status: str
```

**Key rules:**
- Use `total=False` — nodes write only their owned fields
- Custom reducers via `Annotated[type, reducer_fn]` for append-with-limit fields
- Candidates stored as plain dicts `{id, content, document_name}`, never ORM

## Six Core Modules

### 1. Permission Gate (Deterministic, No LLM)

```python
def permission_gate_node(state: RAGState) -> dict:
    if state["role"] != "manager" and any(
        kw in state["question"] for kw in MANAGER_KEYWORDS
    ):
        return {"permission_denied": True, "answer": PERMISSION_DENIED_MESSAGE}
    return {"permission_denied": False}
```

Pattern: keyword matching or rule engine, zero LLM cost, short-circuits to END.

### 2. Intent Gateway (Two-Stage Sub-graph)

**Stage 1 — Classify**: `action ∈ {refuse, clarify, direct, search}`
**Stage 2 — Slot extraction**: task, entities, time, risk_note, search_keywords

```python
class ClassifyResult(BaseModel):
    action: Literal["refuse", "clarify", "direct", "search"]
    risk_reason: str = ""

class SlotResult(BaseModel):
    task: str = ""
    entities: list[str] = []
    search_keywords: list[str] = []
    clarification_question: str = ""
```

**Critical**: Use `method="function_calling"` for structured output when targeting DeepSeek or models without `json_schema` support.

**Degradation**: Classify failure → default to `search` + fallback slots; Slot failure → use raw question as keyword.

### 3. Query Rewrite (Single Structured Call)

Produces three outputs in one LLM call:
- `rewritten_question`: Formalized, coreference-resolved
- `sub_questions`: Decomposed compound questions (0-3)
- `keywords`: Optimized retrieval terms (2-6)

Consumes short-term memory (`conversation_history`, `last_task_status`) for coreference resolution and clarification continuation.

**Merge strategy**: Rewrite keywords + slot keywords → deduplicated first-hop query string; fallback to raw question if both empty.

### 4. Multi-Hop Retrieval Loop

```
search_plan → embed → search → [route_search] → hop_reason → [route_after_hop] → embed...
                                      ↓ (finished/exhausted/max_hops)
                                   rerank
```

**Loop control parameters:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `HOP_MAX_HOPS` | 3 | Total retrieval iterations cap |
| `HOP_TOP_K_PER_HOP` | 5 | Candidates per hop |
| `HOP_TOP_K_CONTEXT` | 6 | Final context chunks after rerank |
| `VERIFY_MISSING_STREAK` | 2 | Same missing aspects → hard stop |

**Hop reasoner**: LLM evaluates accumulated evidence, decides `finished: bool` and generates next-hop query with self-contained keywords (no pronouns).

**State reset on new turn**: `search_plan_node` must reset all loop fields — Checkpointer restores previous terminal state, without reset the second query's loop gets skipped.

### 5. Evidence Verification (Four Conditions)

```
① Coverage: All sub-questions addressed by evidence
② Authority: Sources are official documents/manuals
③ Consistency: Citations support conclusion without contradiction
④ Safety: No permission/sensitive content leaks
```

**Output**: `VerifyResult(sufficient, missing_aspects, next_query, next_keywords)`

**Hard-stop conditions** (any triggers termination):
- `hop_count >= HOP_MAX_HOPS`
- Same `missing_aspects` repeated `VERIFY_MISSING_STREAK` times
- `no_new_hits == True` (zero new candidates from last search)

On hard stop: set `evidence_status = "uncertain"`, inject hedging instructions into answer prompt.

**Degradation**: Verification exception → treat as sufficient (fail-open, avoid infinite loops).

### 6. Hybrid Search + Reranking

**Search**: Vector (ChromaDB ANN) + Keyword (SQLite FTS5) → merge & deduplicate by ID

**Rerank**: Read stored embeddings from vector DB, compute cosine similarity against query vector, sort descending, truncate to `HOP_TOP_K_CONTEXT`.

```python
def rerank_node(state: RAGState) -> dict:
    embeddings = vector_store.get_embeddings([c["id"] for c in hits])
    ranked = sorted(hits, key=lambda c: _cosine(state["embedding"], embeddings[c["id"]]), reverse=True)
    return {"ranked": ranked[:HOP_TOP_K_CONTEXT]}
```

## Short-Term Memory Pattern

- **Checkpointer**: `MemorySaver` (in-process, zero persistence overhead)
- **Thread ID**: Frontend-generated `conversation_id` (sessionStorage lifecycle)
- **History reducer**: Keep last N turns, truncate answer snippets

```python
HISTORY_TURNS = 3
_ANSWER_SNIPPET_LEN = 300

def _keep_recent_history(old: list, new: list) -> list:
    return ((old or []) + (new or []))[-HISTORY_TURNS:]
```

**Write-back**: All terminal branches call `_remember_turn()` via `graph.update_state(config, ..., as_node="answer")`.

## Graph Lifecycle: Singleton + Request Injection

```python
_GRAPH_CACHE: dict = {}

def get_compiled_graph(api_key: str, model=None):
    key = (api_key, id(model) if model else None)
    if key not in _GRAPH_CACHE:
        _GRAPH_CACHE[key] = build_rag_graph(api_key, model=model)
    return _GRAPH_CACHE[key]

# Request-scoped DB injected via config, NOT captured in closures
config = {"configurable": {"db": db_session, "thread_id": conversation_id}}
final = graph.invoke({"question": q, "role": role}, config=config)
```

**Why**: Compiled graph is stateless (state lives in Checkpointer per thread). Nodes access request-scoped resources via `config["configurable"]`, never closure capture.

## Degradation Strategy Summary

| Module | Failure Mode | Fallback |
|--------|-------------|----------|
| Intent classify | LLM error | `action="search"` + raw question as keyword |
| Slot extraction | LLM error | `task="未知"` + raw question as keyword |
| Query rewrite | LLM error | Original question, no sub-questions |
| Hop reasoning | LLM error | Consume queue head sequentially; empty → finish |
| Evidence verify | LLM error | Treat as sufficient (fail-open) |
| Hybrid search | Both paths empty | Return latest chunks as last resort |
| Rerank | Embedding missing | Score = -1.0 (sort to bottom) |

## Configuration Best Practices

- All tunable parameters in a central `config.py`, overridable via environment variables
- Loop bounds (max hops, streak limits) as integers with sensible defaults
- Feature flags (e.g., `USE_BGE_MODEL`) for test environments to skip heavy model loading

## Testing Patterns

- Inject `model` parameter into `build_*` functions for mock replacement
- Mock `with_structured_output` to route by schema class → return canned results
- Use `isolated_vector_store` fixture redirecting ChromaDB to tmp directory
- Set `USE_BGE_MODEL=false` in test env to avoid loading 400MB+ models
- Test each node function independently with crafted state dicts

## Default Retrieval Tools (MUST USE)

When implementing the retrieval layer, **directly use** the tool files in [tools/](tools/) directory:

| File | Deploy To | Responsibility |
|------|-----------|----------------|
| [tools/embedding.py](tools/embedding.py) | `app/services/embedding.py` | Text → 768d vector (BGE model, thread-safe singleton) |
| [tools/vector_store.py](tools/vector_store.py) | `app/services/vector_store.py` | ChromaDB CRUD + ANN query (RLock serialized) |
| [tools/searcher.py](tools/searcher.py) | `app/services/retrieval/searcher.py` | Hybrid search: FTS5 + vector, permission filtering |
| [tools/search_tool.py](tools/search_tool.py) | `app/services/retrieval/search_tool.py` | LangChain @tool wrapper with 5-element description |

### Tool Call Chain

```
User query (text)
    │
    ▼
embed_query(text) → list[float]     # embedding.py: 文本→768维向量
    │
    ▼
hybrid_search(db, vec, text, top_k)  # searcher.py: 向量+关键词双路召回
    │
    ├─ keyword_search (FTS5 MATCH)   # 精确匹配，优先占位
    ├─ vector_search (ChromaDB ANN)  # 语义召回，补充覆盖
    └─ merge & deduplicate           # dict.fromkeys 保序去重
    │
    ▼
list[Chunk] → convert to pure dict   # {id, content, document_name}
```

### Critical Rules for Tool Usage

1. **向量在工具内部生成** — `embed_query(text)` 在 search_node 内部调用，绝不将 vector 作为 LLM tool 参数
2. **LLM 只接触文本参数** — `@tool` 的 `query: str` 是唯一入参，工具描述遵循 5 要素规范
3. **ORM 不进 State** — `hybrid_search` 返回 Chunk ORM 对象后，必须立即转为纯数据 dict
4. **DB 从 Config 注入** — search_node 通过 `config["configurable"]["db"]` 获取 Session，不闭包捕获
5. **权限过滤统一生效** — `exclude_doc_ids` 在 keyword/vector/fallback 三条路径都过滤

### search_node Standard Implementation

```python
def search_node(state: RAGState, config: RunnableConfig) -> dict:
    """Agent 图中的检索节点 — 必须按此模式实现"""
    db = config["configurable"]["db"]  # 请求级注入
    exclude = manager_doc_ids(db) if state["role"] != ROLE_MANAGER else None

    # 调用 tools/searcher.py 的 hybrid_search
    hits = hybrid_search(db, state["embedding"], state["current_query"],
                         top_k=HOP_TOP_K_PER_HOP, exclude_doc_ids=exclude)

    # ORM → 纯数据 dict（checkpoint 序列化要求）+ chunk_id 去重
    seen = {c["id"] for c in state.get("all_hits", [])}
    new_hits = [{"id": c.id, "content": c.content,
                 "document_name": c.document.filename if c.document else "未知"}
                for c in hits if c.id not in seen]

    return {"all_hits": state.get("all_hits", []) + new_hits,
            "hop_count": state.get("hop_count", 0) + 1,
            "no_new_hits": not new_hits}
```

### @tool Description Template (5 Elements)

When wrapping `hybrid_search` as a LangChain Tool, the docstring MUST cover:

```python
@tool
def search_knowledge_base(query: str) -> dict:
    """[1.功能] 在企业知识库中检索相关资料片段。
    知识库涵盖：{列出具体主题域}。

    [2.正向触发] 何时调用：问题涉及上述主题且需事实性资料时。
    [3.负向触发] 何时不调用：闲聊、常识、无关问题直接回答，不调用。

    [4.参数规范] query：提炼为 2~6 个核心关键词，不照搬原问题。
    正确："员工 温暖基金 发放标准"
    错误："我想问一下公司的温暖基金是怎么发放的呀？"

    [5.返回语义] {"results": [...], "count": int}
    count=0 表示无相关内容，必须诚实告知用户，严禁编造。
    """
```

## Additional Resources

- For complete state graph code patterns, see [examples.md](examples.md)
- For architecture decisions and trade-offs, see [architecture-reference.md](architecture-reference.md)
