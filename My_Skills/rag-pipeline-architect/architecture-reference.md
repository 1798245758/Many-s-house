# Architecture Reference: Design Decisions & Trade-offs

## Why LangGraph StateGraph over Pure LCEL

| Dimension | Pure LCEL (RunnableSequence) | LangGraph StateGraph |
|-----------|------------------------------|---------------------|
| Flow control | Linear, no branching | Conditional edges, cycles |
| State management | `RunnablePassthrough.assign` accumulation | TypedDict with reducers |
| Multi-hop loops | Impossible without recursion | Native cycle support |
| Checkpointing | None | Built-in MemorySaver/SqliteSaver |
| Observability | Must instrument each step | State snapshots at every node |
| Complexity | Lower for simple pipelines | Higher setup, pays off with branching |

**Decision**: Use StateGraph when you need conditional routing, cycles (multi-hop), or short-term memory. Use pure LCEL for simple linear chains (embed → search → generate).

## Intent-First vs Rewrite-First Architecture

### Rejected: Rewrite → Intent (coupled)
- Intent classification consumes rewritten question
- Every request pays rewrite LLM cost even if refused
- Data coupling: rewrite failure cascades to intent

### Adopted: Intent → Rewrite (decoupled)
- Intent classifies on raw question (zero dependency)
- `refuse`/`clarify` short-circuit before rewrite (saves one LLM call)
- Rewrite consumes intent slots for keyword merging (unidirectional data flow)
- Independent testability of each module

**Trade-off**: Intent sees unpolished user input (typos, abbreviations). Mitigated by: classify prompt handles informal language; rewrite still runs for `search`/`direct` paths.

## Multi-Hop vs Single-Shot Retrieval

### Single-shot limitations:
- Compound questions need multiple knowledge pieces
- One query string can't cover all aspects
- No mechanism to detect incomplete evidence

### Multi-hop design choices:

**Queue-driven**: Sub-questions from rewrite form initial queue; hop_reasoner generates additional queries based on accumulated evidence.

**Termination guarantees** (prevent infinite loops):
1. Hard cap: `HOP_MAX_HOPS` (default 3)
2. Evidence-driven: LLM decides `finished=true`
3. Queue exhaustion: No remaining sub-questions
4. Diminishing returns: `no_new_hits` detection

**Candidate accumulation**: Each hop's results append to `all_hits` with chunk_id deduplication. Rerank operates on the full accumulated set, not per-hop.

## Evidence Verification: Fail-Open vs Fail-Closed

**Decision: Fail-open** (exception → treat as sufficient)

Rationale:
- Verification is a quality enhancement, not a security gate
- Fail-closed + verification bug = system permanently stuck in retry loop
- Hard-stop conditions already prevent infinite retries
- Users prefer a hedged answer over no answer

**Contrast with permission gate**: Fail-closed (exception → deny). Security gates must never fail-open.

## Hybrid Search Strategy

### Why both vector + keyword?

| Method | Strengths | Weaknesses |
|--------|-----------|------------|
| Vector (ANN) | Semantic similarity, handles paraphrasing | Misses exact terms (IDs, names) |
| Keyword (FTS5) | Exact match, entity names, codes | No semantic understanding |
| Hybrid | Covers both dimensions | Slightly higher latency |

**Merge strategy**: `list(dict.fromkeys(keyword_ids + vector_ids))[:top_k]`
- Keyword results first (higher precision for exact matches)
- Vector results fill remaining slots (semantic recall)
- `dict.fromkeys` preserves insertion order while deduplicating

### Reranking: Why Post-Hoc Cosine?

Hybrid search returns candidates ranked by their respective methods (FTS5 BM25 score vs ANN distance). These scores are incomparable. Reranking with a unified cosine similarity against the original query vector provides a consistent ordering across both retrieval paths.

**Implementation**: Read stored embeddings from ChromaDB by chunk ID → compute cosine with query vector → sort descending → truncate.

## Short-Term Memory Architecture

### MemorySaver (in-process) vs SqliteSaver (persistent)

**For Q&A pipeline**: MemorySaver
- Conversations are ephemeral (browser tab lifecycle)
- Zero disk I/O overhead
- Automatic cleanup on process restart
- Sufficient for single-user local deployment

**For async tasks** (document ingestion): SqliteSaver
- Tasks survive process restarts
- Enable checkpoint-resume (`invoke(None, config)`)
- Progress queryable from separate SSE endpoint

### Conversation History Design

```
Format: "问：{question}\n答：{answer[:300]}"
Window: Last 3 turns
Reducer: Append + truncate (custom Annotated reducer)
```

**Why truncate answers**: Full answers can be 1000+ chars. In prompt context, only the gist matters for coreference resolution. 300 chars captures the key conclusion without blowing up token count.

### Cross-Turn State Continuity

`last_task_status` enables:
- Clarification continuation: "等待用户澄清：{question}" → rewrite merges context
- Error awareness: "已回答但证据不足" → next rewrite can adjust strategy
- Completion signal: "已完成回答" → normal flow

## Singleton Graph + Request Injection Pattern

### Problem: Why not build graph per request?
- Graph compilation involves node registration, edge wiring, checkpointer setup
- ~10-50ms overhead per compilation
- Under load: significant cumulative cost

### Problem: Why not capture DB in closure?
- SQLAlchemy Session is request-scoped (connection pool)
- Closure capture → session leak, stale connections
- Multi-threaded access to same session → corruption

### Solution: Configurable injection
```python
# Build once
graph = build_rag_graph(api_key)  # No request-scoped deps

# Per-request injection
config = {"configurable": {"db": request_db, "thread_id": conv_id}}
result = graph.invoke(state, config=config)

# Node reads from config
def search_node(state, config: RunnableConfig):
    db = config["configurable"]["db"]  # Request-scoped, safe
```

## Structured Output: function_calling vs json_schema

**Constraint**: DeepSeek (and many OpenAI-compatible APIs) don't support `response_format: json_schema`.

**Solution**: `model.with_structured_output(Schema, method="function_calling")`
- Uses tool/function calling protocol
- Model generates arguments matching the Pydantic schema
- LangChain parses and validates automatically
- On validation failure: raises exception → triggers degradation path

**Schema design tips**:
- Use `Literal` for enum fields (invalid values → parse failure → fallback)
- Use `Field(default=...)` for optional fields (reduces parse failures)
- Keep schemas flat (nested models increase failure rate with smaller LLMs)

## Ingestion Pipeline: Offline Architecture

```
Upload → Parse (LangChain Loaders) → Clean → Chunk (500-1000 tokens, 100 overlap)
       → Embed (BGE bge-base-zh-v1.5, 768d) → Write (SQLite + ChromaDB + FTS5)
```

### Chunking Strategy

Separator priority (coarse → fine, progressive degradation):
1. `\n\s*\n` — Paragraph boundaries
2. `\n` — Line boundaries
3. `(?<=[。！？!?；;…])` — Sentence boundaries (zero-width, preserves punctuation)
4. `(?<=[，,、])` — Clause boundaries
5. Character-level hard cut (last resort)

**Parameters**: Target 500-1000 tokens per chunk, 100 token overlap for context continuity.

### Vector Store Abstraction

```python
# vector_store.py — ChromaDB PersistentClient wrapper
# - collection = "chunks", IDs aligned with SQLite chunks.id
# - cosine distance metric
# - RLock for operation-level serialization (ChromaDB not thread-safe)
# - get_embeddings(ids) for batch vector retrieval (reranking)
```

**Why aligned IDs**: Enables cross-store joins — search ChromaDB for similar vectors, then fetch full content from SQLite by same ID.
