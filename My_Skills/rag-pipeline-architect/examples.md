# Code Examples: RAG Pipeline Implementation Patterns

## Complete Graph Construction

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

def build_rag_graph(api_key: str, model=None):
    """Build and compile the RAG state graph."""
    if model is None:
        model = init_chat_model(
            "deepseek-chat",
            model_provider="openai",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0.7,
        )

    graph = StateGraph(RAGState)

    # Register nodes
    graph.add_node("permission_gate", permission_gate_node)
    graph.add_node("intent", intent_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("direct", direct_node)
    graph.add_node("search_plan", search_plan_node)
    graph.add_node("embed", embed_node)
    graph.add_node("search", search_node)
    graph.add_node("hop_reason", hop_reason_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("verify", verify_node)
    graph.add_node("context", context_node)
    graph.add_node("answer", answer_node)

    # Wire edges
    graph.add_edge(START, "permission_gate")
    graph.add_conditional_edges(
        "permission_gate", route_permission,
        {"end": END, "intent": "intent"}
    )
    graph.add_conditional_edges(
        "intent", route_action,
        {"end": END, "rewrite": "rewrite"}
    )
    graph.add_conditional_edges(
        "rewrite", route_after_rewrite,
        {"direct": "direct", "search_plan": "search_plan"}
    )
    graph.add_edge("direct", END)
    graph.add_edge("search_plan", "embed")
    graph.add_edge("embed", "search")
    graph.add_conditional_edges(
        "search", route_search,
        {"hop": "hop_reason", "rerank": "rerank"}
    )
    graph.add_conditional_edges(
        "hop_reason", route_after_hop,
        {"embed": "embed", "rerank": "rerank"}
    )
    graph.add_conditional_edges(
        "rerank", route_hits,
        {"end": END, "verify": "verify"}
    )
    graph.add_conditional_edges(
        "verify", route_verify,
        {"context": "context", "embed": "embed"}
    )
    graph.add_edge("context", "answer")
    graph.add_edge("answer", END)

    return graph.compile(checkpointer=MemorySaver())
```

## Intent Gateway: Two-Stage Sub-graph

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate

class ClassifyResult(BaseModel):
    action: Literal["refuse", "clarify", "direct", "search"] = Field(
        description="Action type"
    )
    risk_reason: str = ""

class SlotResult(BaseModel):
    task: str = ""
    entities: list[str] = []
    time: str = ""
    risk_note: str = ""
    search_keywords: list[str] = []
    clarification_question: str = ""

def build_intent_gate(api_key: str, model=None):
    """Two-stage intent classification sub-graph."""
    if model is None:
        model = init_chat_model("deepseek-chat", model_provider="openai",
                                base_url="https://api.deepseek.com",
                                api_key=api_key, temperature=0.3)

    # CRITICAL: method="function_calling" for DeepSeek compatibility
    classify_chain = (
        ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
        | model.with_structured_output(ClassifyResult, method="function_calling")
    )
    slot_chain = (
        ChatPromptTemplate.from_template(SLOT_PROMPT)
        | model.with_structured_output(SlotResult, method="function_calling")
    )

    def classify_node(state):
        try:
            cls = classify_chain.invoke({"question": state["question"]})
            return {"action": cls.action, "risk_reason": cls.risk_reason}
        except Exception as e:
            # Degradation: default to search with fallback slots
            return {"action": "search", "risk_reason": "",
                    "slots": SlotResult(task="未知", search_keywords=[state["question"]])}

    def slot_node(state):
        if state.get("slots") is not None:
            return {}  # Already set by classify degradation
        try:
            slots = slot_chain.invoke({"question": state["question"],
                                       "action": state["action"]})
            return {"slots": slots}
        except Exception:
            return {"slots": SlotResult(task="未知", search_keywords=[state["question"]])}

    # Build sub-graph
    graph = StateGraph(IntentState)
    graph.add_node("classify", classify_node)
    graph.add_node("slot", slot_node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify",
        lambda s: "end" if s.get("action") == "refuse" else "slot",
        {"end": END, "slot": "slot"})
    graph.add_edge("slot", END)
    compiled = graph.compile()

    def intent_gate(question: str) -> dict:
        out = compiled.invoke({"question": question})
        return {"action": out["action"], "risk_reason": out.get("risk_reason", ""),
                "slots": out.get("slots"), "question": question}

    return intent_gate
```

## Multi-Hop Reasoner with Degradation

```python
class HopDecision(BaseModel):
    finished: bool = Field(description="Whether evidence is sufficient")
    next_query: str = Field(default="", description="Next hop query (self-contained)")
    next_keywords: list[str] = Field(default=[], description="Next hop keywords")

def build_hop_reasoner(api_key: str, model=None):
    """Multi-hop reasoning: decide whether to continue or finish."""
    hop_chain = (
        ChatPromptTemplate.from_template(HOP_PROMPT)
        | model.with_structured_output(HopDecision, method="function_calling")
    )

    def hop_reasoner(question: str, evidence: str, remaining: list) -> HopDecision:
        try:
            return hop_chain.invoke({
                "question": question,
                "evidence": evidence or "(none)",
                "remaining": "; ".join(remaining) if remaining else "(none)",
            })
        except Exception:
            # Degradation: consume queue head; empty queue → finish
            if remaining:
                return HopDecision(finished=False, next_query=remaining[0],
                                   next_keywords=[remaining[0]])
            return HopDecision(finished=True)

    return hop_reasoner
```

## Evidence Verifier with Hard-Stop Logic

```python
class VerifyResult(BaseModel):
    sufficient: bool
    missing_aspects: list[str] = Field(default=[])
    next_query: str = Field(default="")
    next_keywords: list[str] = Field(default=[])

def verify_node(state: RAGState) -> dict:
    """Four-condition evidence verification with hard-stop detection."""
    verifier = build_evidence_verifier(api_key, model=model)
    evidence = "\n".join(
        f"[Source: {c['document_name']}]\n{c['content'][:200]}"
        for c in state["ranked"]
    )
    result = verifier(state["rewritten_question"],
                      state.get("sub_questions") or [], evidence)

    if result.sufficient:
        return {"evidence_status": "verified", "verify_retrying": False}

    # Hard-stop detection
    missing = sorted(result.missing_aspects)
    streak = (state.get("missing_streak", 0) + 1
              if missing == sorted(state.get("last_missing") or [])
              else 1)
    hard_stop = (
        state["hop_count"] >= HOP_MAX_HOPS          # Cap reached
        or streak >= VERIFY_MISSING_STREAK           # Same gaps repeating
        or state.get("no_new_hits")                  # No new knowledge
    )

    if hard_stop:
        return {"evidence_status": "uncertain",
                "missing_aspects": result.missing_aspects,
                "verify_retrying": False}

    # Continue: generate alternative query for next iteration
    next_query = (" ".join(result.next_keywords)
                  if result.next_keywords else result.next_query)
    return {"current_query": next_query or state["rewritten_question"],
            "verify_retrying": True,
            "missing_streak": streak, "last_missing": missing}
```

## Search Plan: State Reset Pattern

```python
def search_plan_node(state: RAGState) -> dict:
    """Initialize multi-hop loop AND reset previous turn's residual state.
    
    CRITICAL: Checkpointer restores previous terminal state.
    Without reset, second query's loop gets skipped because
    loop_finished/evidence_status still hold last turn's values.
    """
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
```

## Hybrid Search with Permission Filtering

```python
def hybrid_search(db, query_vec, query_text, top_k=10, exclude_doc_ids=None):
    """Vector + Keyword hybrid with role-based filtering."""
    excluded = set(exclude_doc_ids or [])

    keyword_ids = keyword_search(db, query_text, top_k=top_k)
    vector_results = vector_search(db, query_vec, top_k=top_k)
    vector_ids = [cid for cid, _ in vector_results]

    # Merge: keyword first (precision), vector fills (recall)
    all_ids = list(dict.fromkeys(keyword_ids + vector_ids))[:top_k]

    if not all_ids:
        # Last resort: recent chunks
        chunks = db.query(Chunk).order_by(Chunk.id.desc()).limit(top_k * 3).all()
        return [c for c in chunks if c.document_id not in excluded][:top_k]

    chunks = db.query(Chunk).filter(Chunk.id.in_(all_ids)).all()
    if excluded:
        chunks = [c for c in chunks if c.document_id not in excluded]
    # Preserve merge order
    id_order = {cid: i for i, cid in enumerate(all_ids)}
    chunks.sort(key=lambda c: id_order.get(c.id, len(all_ids)))
    return chunks[:top_k]
```

## Short-Term Memory Write-Back

```python
HISTORY_TURNS = 3
_ANSWER_SNIPPET_LEN = 300

def _keep_recent_history(old: list, new: list) -> list:
    """Custom reducer: append new entries, keep only last N."""
    return ((old or []) + (new or []))[-HISTORY_TURNS:]

def _remember_turn(graph, config: dict, question: str, resp):
    """All terminal branches converge here to persist turn memory."""
    entry = f"Q: {question}\nA: {resp.answer[:_ANSWER_SNIPPET_LEN]}"
    graph.update_state(config, {
        "conversation_history": [entry],
        "last_task_status": _task_status(resp),
    }, as_node="answer")  # Virtual write, no node execution triggered
```

## Answer Node with Hedging (Uncertain Evidence)

```python
def answer_node(state: RAGState) -> dict:
    """Generate answer; inject hedging instructions when evidence is uncertain."""
    extra = ""
    if state.get("evidence_status") == "uncertain":
        missing = ", ".join(state.get("missing_aspects") or []) or "key information"
        extra = (f"\n\nNote: Evidence is insufficient (missing: {missing}). "
                 "Retrieval terminated due to objective constraints. "
                 "Answer with uncertainty, state evidence limitations, "
                 "do NOT give definitive conclusions.")

    answer_chain = (
        RunnableLambda(lambda s: {"context": s["context"],
                                  "question": s["rewritten_question"],
                                  "extra": extra})
        | ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}{extra}"),
        ])
        | model
        | StrOutputParser()
    )
    return {"answer": answer_chain.invoke(state)}
```

## Testing Pattern: Mock Model Injection

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_model():
    """Mock that routes with_structured_output by schema class."""
    model = MagicMock()
    
    def _structured(schema, **kwargs):
        chain = MagicMock()
        if schema == ClassifyResult:
            chain.invoke.return_value = ClassifyResult(action="search")
        elif schema == SlotResult:
            chain.invoke.return_value = SlotResult(
                task="制度咨询", search_keywords=["员工", "福利"])
        elif schema == RewriteResult:
            chain.invoke.return_value = RewriteResult(
                rewritten_question="员工福利有哪些",
                sub_questions=[], keywords=["员工福利"])
        elif schema == HopDecision:
            chain.invoke.return_value = HopDecision(finished=True)
        elif schema == VerifyResult:
            chain.invoke.return_value = VerifyResult(sufficient=True)
        return chain
    
    model.with_structured_output = _structured
    # For direct prompt|model|parser chains
    model.__or__ = lambda self, other: MagicMock()
    return model

def test_search_flow(mock_model, db_session):
    """Test complete search path with mocked LLM."""
    result = rag_chain_query(
        db=db_session, query_text="员工福利",
        api_key="test-key", model=mock_model, role="employee"
    )
    assert result.response_type == "answer"
    assert result.answer != ""
```
