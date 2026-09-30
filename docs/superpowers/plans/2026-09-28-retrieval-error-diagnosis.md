# RAG 检索错误分析与统一诊断 Agent 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 RAG 检索/问答链路增加"每轮多步轨迹 + 三级错误状态机 + 点击日志触发错误分析"，并统一到现有诊断 Agent（经理专用）。

**Architecture:** 方案 B——检索图内 `trace` 通道逐节点累积步记录（覆盖语义、每轮入口重置），收口按最高级分级并落新表 `QueryTrace`；诊断 Agent 泛化为按证据类型（task/query）分流，复用 L0–L3 多轮上下文；前端诊断页新增"检索/问答错误"标签。

**Tech Stack:** Python 3.12 / FastAPI / LangGraph / SQLAlchemy / pytest；React + Vite。

---

## 参考约定（所有任务通用）

- 后端测试运行目录 `backend/`，命令 `py -m pytest <目标> -v`，环境 `USE_BGE_MODEL=false`（见 `backend/.env`）。
- 全量回归：`py -m pytest tests/ -q`，预期无新增失败。
- 步记录 level 取值仅 `normal|degraded|error`；code 取 §3 清单（E1–E5/W1–W6/N1–N5）。
- `trace` 是**普通字段**（非 reducer）：节点返回 `state.get("trace",[]) + [rec]` 覆盖累积；入口 `permission_gate` 返回 `{"trace": []}` 重置。
- 提交信息用 `feat:`/`test:` 前缀，频繁小步提交。

## 文件结构（新增/修改一览）

- 改 `backend/app/services/retrieval/rewrite_chain.py`：`RewriteResult` 加 `degraded`
- 改 `backend/app/services/retrieval/hop_chain.py`：`HopDecision` 加 `degraded`
- 改 `backend/app/services/retrieval/verify_chain.py`：`VerifyResult` 加 `degraded`
- 改 `backend/app/services/retrieval/intent_chain.py`：`IntentState`/gate 回传 `degraded`
- 改 `backend/app/services/retrieval/agent_chain.py`：`RAGState.trace`、`_step`、各节点埋点、`_classify_trace`、`_record_trace`、收口接入
- 新 `backend/app/models/query_trace.py`：`QueryTrace` 模型
- 改 `backend/app/main.py`：`init_db` 覆盖新表（create_all 已含，仅需 import 保证注册）
- 改 `backend/app/routers/query.py`：整轮失败落 E5
- 改 `backend/app/services/diagnosis/agent.py`：`collect_query_evidence`、`RETRIEVAL_SYSTEM_PROMPT`、`analyze` 分流
- 改 `backend/app/services/diagnosis/context.py`：主体泛化 `(subject_type, subject_id)`
- 改 `backend/app/schemas/diagnosis.py`：`QueryAnalyzeRequest`、`DiagnosisChatRequest` 增字段
- 改 `backend/app/routers/diagnosis.py`：query-errors / query-traces / analyze-query / chat 分流
- 新 `backend/tests/test_query_trace.py`、`backend/tests/test_diagnosis_query.py`
- 改 `frontend/src/api/diagnosis.js`、`frontend/src/pages/Diagnosis/index.jsx`

---

## Task 1: 各链回传 degraded 标记

**Files:**
- Modify: `backend/app/services/retrieval/rewrite_chain.py`
- Modify: `backend/app/services/retrieval/hop_chain.py`
- Modify: `backend/app/services/retrieval/verify_chain.py`
- Modify: `backend/app/services/retrieval/intent_chain.py`
- Test: `backend/tests/test_degraded_flag.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_degraded_flag.py`：
```python
from unittest.mock import MagicMock
from app.services.retrieval.rewrite_chain import build_query_rewriter
from app.services.retrieval.hop_chain import build_hop_reasoner
from app.services.retrieval.verify_chain import build_evidence_verifier
from app.services.retrieval.intent_chain import build_intent_gate


def test_rewrite_degraded_on_llm_error():
    m = MagicMock()
    m.with_structured_output.return_value.invoke.side_effect = RuntimeError("x")
    out = build_query_rewriter("k", model=m)("原问题", history=[], task_status="none")
    assert out.degraded is True


def test_hop_degraded_on_llm_error():
    m = MagicMock()
    m.with_structured_output.return_value.invoke.side_effect = RuntimeError("x")
    out = build_hop_reasoner("k", model=m)("q", "evidence", ["子问题"])
    assert out.degraded is True


def test_verify_degraded_on_llm_error():
    m = MagicMock()
    m.with_structured_output.return_value.invoke.side_effect = RuntimeError("x")
    out = build_evidence_verifier("k", model=m)("q", ["子问题"], "evidence")
    assert out.degraded is True and out.sufficient is True


def test_intent_degraded_on_llm_error():
    m = MagicMock()
    m.with_structured_output.return_value.invoke.side_effect = RuntimeError("x")
    out = build_intent_gate("k", model=m)("问")
    assert out["degraded"] is True and out["action"] == "search"
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_degraded_flag.py -v`
Expected: FAIL（`degraded` 字段/键不存在）

- [ ] **Step 3: 改 rewrite_chain.py**

`RewriteResult` 加字段：
```python
class RewriteResult(BaseModel):
    rewritten_question: str = Field(default="", description="补全指代后的完整问题")
    search_keywords: list[str] = Field(default=[], description="改写链额外补充的检索关键词")
    degraded: bool = Field(default=False, description="LLM 失败走兜底")
```
`fallback_rewrite` 置 `degraded=True`：
```python
def fallback_rewrite(question: str, slots: SlotResult | None) -> RewriteResult:
    if slots and slots.search_keywords:
        return RewriteResult(rewritten_question=question,
                             search_keywords=slots.search_keywords, degraded=True)
    return RewriteResult(rewritten_question=question, search_keywords=[question], degraded=True)
```

- [ ] **Step 4: 改 hop_chain.py**

`HopDecision` 加 `degraded: bool = Field(default=False)`；两处 except 兜底 return 加 `degraded=True`：
```python
        except Exception as e:
            logger.warning(f"多跳推理失败，降级处理: {e}")
            if remaining:
                return HopDecision(finished=False, next_query=remaining[0],
                                   next_keywords=[remaining[0]], degraded=True)
            return HopDecision(finished=True, degraded=True)
```

- [ ] **Step 5: 改 verify_chain.py**

`VerifyResult` 加 `degraded: bool = Field(default=False)`；except 兜底：
```python
        except Exception as e:
            logger.warning(f"证据校验失败，降级放行: {e}")
            return VerifyResult(sufficient=True, degraded=True)
```

- [ ] **Step 6: 改 intent_chain.py**

`IntentState` 加 `degraded: bool`；`classify_node`/`slot_node` 兜底置 True；`intent_gate` 回传：
```python
class IntentState(TypedDict, total=False):
    question: str
    action: str
    risk_reason: str
    slots: SlotResult
    degraded: bool
```
classify_node except：`return {"action": "search", "risk_reason": "", "slots": fallback_slots(state["question"]), "degraded": True}`
slot_node except：`return {"slots": fallback_slots(state["question"]), "degraded": True}`
intent_gate 返回 dict 增：`"degraded": out.get("degraded", False)`

- [ ] **Step 7: 运行验证通过**

Run: `py -m pytest tests/test_degraded_flag.py tests/test_rewrite_chain.py tests/test_hop_chain.py tests/test_verify_chain.py tests/test_intent_chain.py -v`
Expected: PASS（含既有链测试不回归）

- [ ] **Step 8: 提交**

```bash
git add backend/app/services/retrieval/rewrite_chain.py backend/app/services/retrieval/hop_chain.py backend/app/services/retrieval/verify_chain.py backend/app/services/retrieval/intent_chain.py backend/tests/test_degraded_flag.py
git commit -m "feat: 各检索链回传 degraded 降级标记"
```

---

## Task 2: RAGState.trace + _step 辅助 + 入口重置

**Files:**
- Modify: `backend/app/services/retrieval/agent_chain.py`
- Test: `backend/tests/test_query_trace.py`

- [ ] **Step 1: 写失败测试（_step 与重置）**

`backend/tests/test_query_trace.py`：
```python
from unittest.mock import MagicMock, patch
from app.services.retrieval.agent_chain import _step, _classify_trace


def test_step_builds_pure_record():
    rec = _step("rerank", "error", code="E1", message="候选集为空", ranked=0)
    assert rec["step"] == "rerank" and rec["level"] == "error"
    assert rec["code"] == "E1" and rec["metrics"] == {"ranked": 0} and rec["ts"]


def test_classify_trace_takes_max_level():
    trace = [_step("intent", "normal", code="N1"),
             _step("verify", "degraded", code="W1"),
             _step("rerank", "error", code="E1")]
    level, code = _classify_trace(trace)
    assert level == "error" and code == "E1"


def test_classify_trace_empty_is_normal():
    assert _classify_trace([]) == ("normal", "")
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_query_trace.py -v`
Expected: FAIL（`_step`/`_classify_trace` 未定义）

- [ ] **Step 3: 在 agent_chain.py 加 trace 字段与辅助**

`RAGState` 末尾加（普通字段，无 Annotated）：
```python
    # 本轮多步诊断轨迹（普通字段：节点覆盖式累积，入口重置；纯数据 dict，可序列化）
    trace: list
```
在 `NO_RESULT_MESSAGE` 常量附近加：
```python
import datetime as _dt

_LEVEL_ORDER = {"normal": 0, "degraded": 1, "error": 2}


def _step(step: str, level: str, code: str = "", message: str = "", **metrics) -> dict:
    return {"step": step, "level": level, "code": code, "message": message,
            "metrics": metrics, "ts": _dt.datetime.now().isoformat(timespec="seconds")}


def _classify_trace(trace: list) -> tuple[str, str]:
    """轮级 = 各步最高级；同级取最后一个。返回 (final_level, final_code)。"""
    best, best_code = "normal", ""
    for s in trace or []:
        if _LEVEL_ORDER.get(s.get("level"), 0) >= _LEVEL_ORDER[best]:
            best = s.get("level", "normal")
            best_code = s.get("code", "")
    return best, best_code
```
> 注：`_classify_trace` 用 `>=` 使同级取最后一个（后发生的更贴近最终态）。

- [ ] **Step 4: permission_gate 入口重置 trace**

`permission_gate_node` 的两个 return 都带 `"trace": []`：
```python
    def permission_gate_node(state: RAGState) -> dict:
        out = _gate(state["question"])
        if out.get("action") == "deny":
            return {"action": "permission_denied", "permission_reason": out["reason"],
                    "permission_detail": out["detail"], "trace": []}
        return {"trace": []}
```

- [ ] **Step 5: 运行验证通过**

Run: `py -m pytest tests/test_query_trace.py -v`
Expected: PASS（前三个用例）

- [ ] **Step 6: 提交**

```bash
git add backend/app/services/retrieval/agent_chain.py backend/tests/test_query_trace.py
git commit -m "feat: RAGState 增加 trace 通道与分级辅助"
```

---

## Task 3: 各节点埋点

**Files:**
- Modify: `backend/app/services/retrieval/agent_chain.py`
- Test: `backend/tests/test_query_trace.py`

- [ ] **Step 1: 追加失败测试（分支产码）**

在 `test_query_trace.py` 追加：
```python
def _graph_state(**kw):
    from app.services.retrieval.agent_chain import build_rag_chain
    return build_rag_chain


def test_empty_candidates_marks_E1():
    from unittest.mock import MagicMock, patch
    from app.services.retrieval import agent_chain as ac
    model = MagicMock()
    # intent→search；rewrite 正常；search 返回命中但 rerank 后为空
    with patch.object(ac, "build_intent_gate") as gate, \
         patch.object(ac, "build_query_rewriter") as rw, \
         patch.object(ac, "hybrid_search") as hs, \
         patch.object(ac, "embed_query") as eq, \
         patch.object(ac, "build_cross_encoder_reranker") as rr:
        gate.return_value = lambda q: {"action": "search", "risk_reason": "",
                                       "slots": MagicMock(task="t", sub_questions=[], search_keywords=["k"]),
                                       "question": q, "degraded": False}
        rw.return_value = lambda q, history, task_status, memories="": MagicMock(
            rewritten_question=q, search_keywords=[], degraded=False)
        eq.return_value = [0.1]
        hs.return_value = [{"id": 1, "doc_id": 1, "content": "x", "filename": "f",
                            "chunk_index": 0, "score": 0.5}]
        rr.return_value = lambda q, cs, top_k: []   # 精排后为空
        g = ac.build_rag_chain("k", model=model, role="员工", manager_doc_ids=set())
        cfg = {"configurable": {"thread_id": "t1", "db": MagicMock()}}
        final = g.invoke({"question": "问", "history": [], "task_status": "none"}, cfg)
    codes = [s["code"] for s in final["trace"]]
    assert "E1" in codes and final["response_type"] == "answer"
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_query_trace.py::test_empty_candidates_marks_E1 -v`
Expected: FAIL（trace 中无 E1）

- [ ] **Step 3: intent_node 埋点**

`intent_node` 在返回前构造 trace 追加（保留原有返回键）：
```python
    def intent_node(state: RAGState) -> dict:
        out = _gate(state["question"])
        action = out.get("action", "search")
        code = {"refuse": "N3", "clarify": "N4", "direct": "N5"}.get(action, "N1")
        rec = _step("intent", "normal", code=code, message=f"action={action}")
        trace = state.get("trace", []) + [rec]
        if out.get("degraded"):
            trace = trace + [_step("intent", "degraded", code="W3", message="意图识别降级")]
        if action == "refuse":
            return {"action": "refuse", "risk_reason": out.get("risk_reason", ""),
                    "current_question": out["question"], "trace": trace}
        if action == "clarify":
            return {"action": "clarify", "clarification_question": out.get("clarification_question", ""),
                    "current_question": out["question"], "trace": trace}
        slots = out.get("slots")
        return {
            "action": "search", "current_question": out["question"],
            "task_type": getattr(slots, "task", "") if slots else "",
            "sub_questions": list(getattr(slots, "sub_questions", [])) if slots else [],
            "slot_keywords": list(getattr(slots, "search_keywords", [])) if slots else [],
            "risk_reason": out.get("risk_reason", ""), "trace": trace,
        }
```

- [ ] **Step 4: rewrite_node 埋点**

在 `rewrite_node` 末尾把 trace 追加（W2 若 degraded）：
```python
        trace = state.get("trace", [])
        if getattr(result, "degraded", False):
            trace = trace + [_step("rewrite", "degraded", code="W2", message="改写降级")]
        return {
            "current_question": result.rewritten_question or state["current_question"],
            "keywords": _dedup((state.get("slot_keywords") or []) + list(result.search_keywords or [])),
            "trace": trace,
        }
```

- [ ] **Step 5: embed_node / search_node 埋点（E2/E3 + candidate_count）**

```python
    def embed_node(state: RAGState) -> dict:
        try:
            return {"query_vector": embed_query(state["current_question"])}
        except Exception as e:
            return {"query_vector": [], "trace": state.get("trace", []) +
                    [_step("embed", "error", code="E2", message=f"向量化异常: {e}")]}

    def search_node(state: RAGState) -> dict:
        if state["action"] == "direct":
            return {"all_hits": [], "retrieved_aspects": set(), "consumed_subs": [],
                    "search_plan": [], "loop_round": 0}
        exclude_ids = None if _role == ROLE_MANAGER else _manager_doc_ids
        try:
            hits = hybrid_search(db=state_db(), query_text=state["current_question"],
                                 query_vector=state["query_vector"], top_k=SEARCH_TOP_K,
                                 keywords=state["keywords"] or None, exclude_doc_ids=exclude_ids)
        except Exception as e:
            hits = []
            trace = state.get("trace", []) + [_step("search", "error", code="E3", message=f"检索异常: {e}")]
            return {"all_hits": [], "retrieved_aspects": set(), "consumed_subs": [],
                    "search_plan": [], "loop_round": 0, "trace": trace}
        trace = state.get("trace", []) + [_step("search", "normal", message="检索",
                                                candidate_count=len(hits))]
        return {"all_hits": _merge_hits(state.get("all_hits"), hits),
                "retrieved_aspects": state.get("retrieved_aspects") | {state["current_question"]},
                "trace": trace}
```
> `state_db()` 已存在（从 config 取 db）。若原 search_node 用的是闭包 `db`，保持其取 db 方式不变，仅包 try/except 与 trace。

- [ ] **Step 6: hop_reason_node 埋点（W4）**

在 `hop_reason_node` 返回 dict 里，若 `decision.degraded` 追加 W4：
```python
        trace = state.get("trace", [])
        if getattr(decision, "degraded", False):
            trace = trace + [_step("hop_reason", "degraded", code="W4", message="多跳推理降级")]
        return {
            "search_plan": list(decision.next_keywords or []),
            "consumed_subs": consumed,
            "loop_round": loop_round + 1,
            "current_question": decision.next_query or (remaining[0] if remaining else state["current_question"]),
            "keywords": _dedup((state.get("slot_keywords") or []) + list(decision.next_keywords or [])),
            "hop_count": hop_count + 1,
            "trace": trace,
        }
```
（其余提前 return 分支同样补 `"trace": state.get("trace", [])` 以免覆盖丢失。）

- [ ] **Step 7: rerank_node 埋点（E1）**

```python
    def rerank_node(state: RAGState) -> dict:
        candidates = _dedup_by_chunk(state.get("all_hits") or [])
        ranked = _reranker(state["question"], candidates, top_k=RERANK_TOP_K)
        trace = state.get("trace", [])
        if not ranked:
            trace = trace + [_step("rerank", "error", code="E1", message="候选集为空", ranked=0)]
        else:
            trace = trace + [_step("rerank", "normal", message="精排", ranked=len(ranked))]
        return {"ranked": ranked, "trace": trace}
```

- [ ] **Step 8: verify_node 埋点（W1/W5/W6）**

在 `verify_node` 内：uncertain 分支追加 W1（附 missing_aspects），degraded 追加 W5，达上限追加 W6；所有 return 带 `trace`。
```python
        trace = state.get("trace", [])
        if getattr(result, "degraded", False):
            trace = trace + [_step("verify", "degraded", code="W5", message="证据校验降级")]
        if result.sufficient:
            if state["hop_count"] >= HOP_MAX_HOPS:
                trace = trace + [_step("verify", "degraded", code="W6", message="达跳数上限")]
            return {"evidence_status": "verified", "trace": trace}
        if state["hop_count"] >= HOP_MAX_HOPS:
            trace = trace + [_step("verify", "degraded", code="W6", message="达跳数上限仍不足"),
                             _step("verify", "degraded", code="W1", message="证据不足硬停",
                                   missing=list(result.missing_aspects or []))]
            return {"evidence_status": "uncertain",
                    "missing_aspects": list(result.missing_aspects or []), "trace": trace}
        trace = trace + [_step("verify", "normal", message="需补充检索")]
        return {"search_plan": list(result.next_keywords or []), "consumed_subs": [],
                "loop_round": 0,
                "current_question": result.next_query or state["question"],
                "keywords": _dedup((state.get("slot_keywords") or []) + list(result.next_keywords or [])),
                "trace": trace}
```
> `evidence_status="uncertain"` 的 W1 也在 answer 硬停分支体现；此处统一在 verify 打点，answer 分支不再重复打 W1。

- [ ] **Step 9: direct_node / answer_node 埋点（N5/N1/E4）**

direct_node 成功返回追加 `_step("answer","normal",code="N5")`，异常追加 `_step("answer","error",code="E4",message=str(e))`。
answer_node 成功追加 `_step("answer","normal",code="N1")`（uncertain 硬停时改 `_step("answer","normal",code="N1",message="uncertain硬停")`，W1 已由 verify 记录）。

- [ ] **Step 10: 运行验证通过**

Run: `py -m pytest tests/test_query_trace.py tests/test_agent_chain.py -v`
Expected: PASS

- [ ] **Step 11: 提交**

```bash
git add backend/app/services/retrieval/agent_chain.py backend/tests/test_query_trace.py
git commit -m "feat: 检索图逐节点埋点记录多步轨迹与错误码"
```

---

## Task 4: QueryTrace 模型

**Files:**
- Create: `backend/app/models/query_trace.py`
- Modify: `backend/app/main.py`
- Test: `backend/tests/test_query_trace.py`

- [ ] **Step 1: 写失败测试**

追加：
```python
def test_query_trace_model_persists(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base
    from app.models.query_trace import QueryTrace
    eng = create_engine(f"sqlite:///{tmp_path/'t.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    s = sessionmaker(bind=eng)()
    s.add(QueryTrace(conversation_id="c1", question="q", final_level="error",
                     final_code="E1", response_type="answer", steps="[]"))
    s.commit()
    row = s.query(QueryTrace).first()
    assert row.final_level == "error" and row.conversation_id == "c1"
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_query_trace.py::test_query_trace_model_persists -v`
Expected: FAIL（模块不存在）

- [ ] **Step 3: 建模型**

`backend/app/models/query_trace.py`：
```python
from sqlalchemy import Column, Integer, String, Text, func
from app.database import Base


class QueryTrace(Base):
    __tablename__ = "query_trace"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, index=True)
    question = Column(Text, nullable=False)
    answer_snippet = Column(Text)
    response_type = Column(String)
    final_level = Column(String, index=True)
    final_code = Column(String)
    steps = Column(Text)
    created_at = Column(String, server_default=func.datetime('now'))
```

- [ ] **Step 4: main.py 确保注册**

在 `from app.models.memory import Memory` 一行后加：
```python
from app.models.query_trace import QueryTrace  # noqa: F401  确保 create_all 建表
```

- [ ] **Step 5: 运行验证通过**

Run: `py -m pytest tests/test_query_trace.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add backend/app/models/query_trace.py backend/app/main.py backend/tests/test_query_trace.py
git commit -m "feat: 新增 QueryTrace 检索轨迹表"
```

---

## Task 5: 收口分级落库 _record_trace

**Files:**
- Modify: `backend/app/services/retrieval/agent_chain.py`
- Test: `backend/tests/test_query_trace.py`

- [ ] **Step 1: 写失败测试**

追加：
```python
def test_record_trace_writes_row():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base
    from app.models.query_trace import QueryTrace
    from app.services.retrieval.agent_chain import _record_trace, _step
    from app.schemas.query import QueryResponse, SourceRef
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    db = sessionmaker(bind=eng)()
    resp = QueryResponse(response_type="answer", answer="答", sources=[],
                         task_status="none", evidence_status="uncertain")
    _record_trace(db, "c1", "问", resp, [_step("rerank", "error", code="E1")])
    row = db.query(QueryTrace).first()
    assert row.final_level == "error" and row.final_code == "E1" and row.conversation_id == "c1"


def test_record_trace_swallows_db_none():
    from app.services.retrieval.agent_chain import _record_trace
    from app.schemas.query import QueryResponse
    resp = QueryResponse(response_type="answer", answer="a", sources=[],
                         task_status="none", evidence_status="verified")
    _record_trace(None, "c", "q", resp, [])  # 不抛异常即通过
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_query_trace.py::test_record_trace_writes_row -v`
Expected: FAIL（`_record_trace` 未定义）

- [ ] **Step 3: 实现 _record_trace**

在 `_classify_trace` 后加（顶部 import json、QueryTrace）：
```python
import json
from app.models.query_trace import QueryTrace


def _record_trace(db, conversation_id, question, resp, trace):
    """落一条 QueryTrace；db 为空或写库异常时静默跳过，绝不阻断问答。"""
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
        logger.warning(f"检索轨迹落库失败，跳过: {e}")
        try:
            db.rollback()
        except Exception:
            pass
```

- [ ] **Step 4: 收口接入所有返回分支**

在 `rag_chain_query` 顶部取 `trace = final.get("trace", [])`（放在 `final = graph.invoke(...)` 之后）。
每个 `_remember_turn(...)` / `_extract_memories(...)` 之后追加 `_record_trace(db, conv_id, query_text, resp, trace)`。
对 permission_denied 分支（无 final）单独构造：
```python
    if gate_out.get("action") == "deny":
        resp = QueryResponse(response_type="permission_denied", answer=PERMISSION_DENY_MESSAGE,
                             sources=[], task_status="none", evidence_status="uncertain")
        _record_trace(db, conversation_id, query_text, resp,
                      [_step("permission_gate", "normal", code="N2", message=gate_out.get("reason", ""))])
        return resp
```

- [ ] **Step 5: 运行验证通过**

Run: `py -m pytest tests/test_query_trace.py tests/test_agent_chain.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add backend/app/services/retrieval/agent_chain.py backend/tests/test_query_trace.py
git commit -m "feat: 问答收口分级并落库 QueryTrace"
```

---

## Task 6: query.py 整轮失败落 E5

**Files:**
- Modify: `backend/app/routers/query.py`
- Test: `backend/tests/test_query_trace.py`

- [ ] **Step 1: 写失败测试**

追加：
```python
def test_query_router_records_E5_on_total_failure(monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.models.query_trace import QueryTrace
    from app.database import get_db
    from app.models.setting import Setting
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    TestS = sessionmaker(bind=eng)()
    TestS.add(Setting(key="api_key", value="sk-test")); TestS.commit()
    import app.routers.query as qr
    monkeypatch.setattr(qr, "generate_answer_via_chain", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(qr, "generate_answer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom2")))
    app.dependency_overrides[get_db] = lambda: TestS
    client = TestClient(app)
    r = client.post("/api/query", json={"question": "问"})
    assert r.json()["code"] == "LLM_ERROR"
    row = TestS.query(QueryTrace).first()
    assert row and row.final_code == "E5" and row.final_level == "error"
    app.dependency_overrides.clear()
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_query_trace.py::test_query_router_records_E5_on_total_failure -v`
Expected: FAIL（无 QueryTrace 行）

- [ ] **Step 3: query.py 落 E5**

在最内层 `except Exception as e:` 返回 LLM_ERROR 之前落库：
```python
from app.models.query_trace import QueryTrace
import json
...
        except Exception as e:
            try:
                db.add(QueryTrace(conversation_id=req.conversation_id or "default",
                                  question=req.question, answer_snippet="",
                                  response_type="error", final_level="error", final_code="E5",
                                  steps=json.dumps([{"step": "graph", "level": "error",
                                                     "code": "E5", "message": str(e)}], ensure_ascii=False)))
                db.commit()
            except Exception:
                db.rollback()
            return ApiResponse(code="LLM_ERROR", message=f"LLM 调用失败: {str(e)} (Chain: {chain_err})")
```

- [ ] **Step 4: 运行验证通过**

Run: `py -m pytest tests/test_query_trace.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/routers/query.py backend/tests/test_query_trace.py
git commit -m "feat: 整轮问答失败落 E5 轨迹"
```

---

## Task 7: 诊断 Agent 采集问答证据 + 分流

**Files:**
- Modify: `backend/app/services/diagnosis/agent.py`
- Test: `backend/tests/test_diagnosis_query.py`

- [ ] **Step 1: 写失败测试**

`backend/tests/test_diagnosis_query.py`：
```python
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models.query_trace import QueryTrace
from app.services.diagnosis import agent


def _db():
    eng = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=eng)
    return sessionmaker(bind=eng)()


def test_collect_query_evidence():
    db = _db()
    db.add(QueryTrace(id=1, conversation_id="c1", question="问", answer_snippet="答",
                      response_type="answer", final_level="error", final_code="E1",
                      steps=json.dumps([{"step": "rerank", "level": "error", "code": "E1",
                                         "message": "候选集为空", "metrics": {"ranked": 0}}])))
    db.commit()
    ev = agent.collect_query_evidence(db, 1)
    assert ev["final_code"] == "E1" and ev["steps"][0]["step"] == "rerank"
    assert agent.collect_query_evidence(db, 999) is None


def test_analyze_query_uses_retrieval_prompt():
    db = _db()
    db.add(QueryTrace(id=2, conversation_id="c", question="q", response_type="answer",
                      final_level="error", final_code="E1", steps="[]"))
    db.commit()
    ev = agent.collect_query_evidence(db, 2)

    class FakeClient:
        def __init__(self): self.sys = ""
        def ask(self, system, user): self.sys = system; return "根因报告"
    c = FakeClient()
    out = agent.analyze(ev, c)
    assert out["analysis"] == "根因报告" and "检索" in c.sys
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_diagnosis_query.py -v`
Expected: FAIL（`collect_query_evidence` 未定义）

- [ ] **Step 3: 实现采集 + 文本化 + prompt + 分流**

在 `agent.py` 加：
```python
import json as _json
from app.models.query_trace import QueryTrace

RETRIEVAL_SYSTEM_PROMPT = (
    "你是企业知识库系统的 RAG 检索链路诊断专家。用户会提供一轮问答的多步状态机轨迹"
    "（意图/改写/检索/精排/多跳/证据校验/回答），以及每步的级别(正常/降级/错误)与错误码。"
    "请基于轨迹分析检索失败或质量降级的根本原因，用 Markdown 输出三部分："
    "## 根因分析、## 关键证据（引用具体步骤/错误码）、## 修复建议。"
    "只依据给定轨迹推理，证据不足时说明还需什么信息，不要编造。"
)


def collect_query_evidence(db, trace_id: int) -> dict | None:
    row = db.query(QueryTrace).filter(QueryTrace.id == trace_id).first()
    if not row:
        return None
    try:
        steps = _json.loads(row.steps or "[]")
    except Exception:
        steps = []
    return {"subject_type": "query", "trace_id": row.id, "conversation_id": row.conversation_id,
            "question": row.question, "answer_snippet": row.answer_snippet or "",
            "response_type": row.response_type or "", "final_level": row.final_level or "normal",
            "final_code": row.final_code or "", "steps": steps}


def _query_evidence_to_text(ev: dict) -> str:
    parts = [f"会话: {ev['conversation_id']}", f"问题: {ev['question']}",
             f"响应类型: {ev['response_type']}，最终级别: {ev['final_level']}（{ev['final_code']}）",
             f"答案摘要: {ev['answer_snippet'][:200]}", "多步轨迹（时间序）:"]
    for s in ev["steps"]:
        parts.append(f"- [{s.get('level')}/{s.get('code','')}] {s.get('step')}: "
                     f"{s.get('message','')} {s.get('metrics') or ''}")
    return "\n".join(parts)
```
把 `analyze` 改为分流（保留原任务逻辑）：
```python
def analyze(evidence: dict, client) -> dict:
    is_query = evidence.get("subject_type") == "query" or "steps" in evidence
    system = RETRIEVAL_SYSTEM_PROMPT if is_query else SYSTEM_PROMPT
    user = _query_evidence_to_text(evidence) if is_query else _evidence_to_text(evidence)
    ident = {"trace_id": evidence.get("trace_id")} if is_query else {"task_id": evidence.get("task_id")}
    try:
        analysis = client.ask(system, user)
    except Exception as e:
        return {**ident, "final_level": evidence.get("final_level", ""),
                "error": evidence.get("error", ""), "analysis": "",
                "llm_error": f"LLM 诊断调用失败: {e}"}
    return {**ident, "final_level": evidence.get("final_level", evidence.get("final_status", "")),
            "error": evidence.get("error", ""), "analysis": analysis, "llm_error": ""}
```
> 注意：任务侧原有返回键 `final_status` 的调用方（routers/diagnosis.py analyze_task）仍需工作——保留 `evidence` 回填与 `llm_error` 判定不变；如测试断言 `final_status`，在返回 dict 中对任务类补 `"final_status": evidence.get("final_status","")`。

- [ ] **Step 4: 运行验证通过（含任务侧不回归）**

Run: `py -m pytest tests/test_diagnosis_query.py tests/test_diagnosis_api.py tests/test_diagnosis_context.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/services/diagnosis/agent.py backend/tests/test_diagnosis_query.py
git commit -m "feat: 诊断 Agent 采集问答轨迹证据并按类型分流"
```

---

## Task 8: 多轮上下文主体泛化

**Files:**
- Modify: `backend/app/services/diagnosis/context.py`
- Test: `backend/tests/test_diagnosis_query.py`

- [ ] **Step 1: 写失败测试**

追加：
```python
def test_ask_query_subject_pins_evidence():
    from app.services.diagnosis import context as ctx
    ev = {"subject_type": "query", "trace_id": 7, "question": "q", "final_level": "error",
          "final_code": "E1", "steps": [{"step": "rerank", "level": "error", "code": "E1",
                                          "message": "候选集为空"}], "response_type": "answer",
          "answer_snippet": "", "conversation_id": "c"}

    class FakeClient:
        def ask(self, system, user): return "ok"
    out = ctx.ask(7, "为什么", ev, FakeClient(), session_id=None, subject_type="query")
    assert out["session_id"] and out["task_id"] == 7 and out["analysis"] == "ok"
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_diagnosis_query.py::test_ask_query_subject_pins_evidence -v`
Expected: FAIL（`ask` 无 `subject_type` 参数）

- [ ] **Step 3: context.py 泛化**

- `ContextManager.__init__(self, subject_type, subject_id, evidence)`：把 `self.task_id` 改名保留为 `self.subject_id`，并加 `self.subject_type`；为兼容既有 router 读取 `mgr.task_id`，保留 `task_id` 属性别名（`@property def task_id(self): return self.subject_id`）。
- `_evidence_text` 按 subject_type 选 `agent._query_evidence_to_text` 或 `agent._evidence_to_text`。
- `ask(subject_id, question, evidence, client, session_id=None, subject_type="task")`：
  - 取/建 mgr 时比较 `mgr.subject_id == subject_id and mgr.subject_type == subject_type`，不一致则重建。
  - 新建：`ContextManager(subject_type, subject_id, evidence)`。
- 返回 dict 保持 `task_id` 键（=subject_id）以兼容前端与既有测试。

- [ ] **Step 4: 运行验证通过（含任务侧多轮不回归）**

Run: `py -m pytest tests/test_diagnosis_query.py tests/test_diagnosis_context.py tests/test_diagnosis_api.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add backend/app/services/diagnosis/context.py backend/tests/test_diagnosis_query.py
git commit -m "feat: 诊断多轮上下文主体泛化为 (subject_type, subject_id)"
```

---

## Task 9: 诊断路由与 schema

**Files:**
- Modify: `backend/app/schemas/diagnosis.py`
- Modify: `backend/app/routers/diagnosis.py`
- Test: `backend/tests/test_diagnosis_query.py`

- [ ] **Step 1: 写失败测试（API）**

追加：
```python
from fastapi.testclient import TestClient
from app.main import app
from app.database import get_db
from app.models.setting import Setting
from app.models.user_role import UserRole
from app.services import role as role_svc
from unittest.mock import patch


def _mgr_db():
    db = _db()
    db.add(Setting(key="api_key", value="sk-test"))
    db.add(UserRole(username="boss", role="manager"))
    db.commit(); return db


def test_query_errors_manager_only():
    db = _mgr_db()
    app.dependency_overrides[get_db] = lambda: db
    with patch.object(role_svc, "get_role", return_value="employee"):
        app.dependency_overrides[role_svc.get_role] = lambda: "employee"
        r = TestClient(app).get("/api/diagnosis/query-errors")
    assert r.json()["code"] == "PERMISSION_DENIED"
    app.dependency_overrides.clear()


def test_query_errors_lists_rows():
    db = _mgr_db()
    db.add(QueryTrace(id=1, conversation_id="c", question="q", response_type="answer",
                      final_level="error", final_code="E1", steps="[]"))
    db.commit()
    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[role_svc.get_role] = lambda: "manager"
    r = TestClient(app).get("/api/diagnosis/query-errors")
    data = r.json()["data"]
    assert any(x["trace_id"] == 1 and x["final_level"] == "error" for x in data)
    app.dependency_overrides.clear()
```

- [ ] **Step 2: 运行验证失败**

Run: `py -m pytest tests/test_diagnosis_query.py -k query_errors -v`
Expected: FAIL（404 / 端点不存在）

- [ ] **Step 3: schema 增字段**

`backend/app/schemas/diagnosis.py` 加：
```python
class QueryAnalyzeRequest(BaseModel):
    trace_id: int
    error_log: str | None = None
```
`DiagnosisChatRequest` 增可选字段：
```python
    subject_type: str = "task"     # task | query
    trace_id: int | None = None
```

- [ ] **Step 4: 路由实现**

`backend/app/routers/diagnosis.py` 加（复用 `_manager_only`、`Setting` 取 key、`DeepSeekClient`）：
```python
from app.models.query_trace import QueryTrace
from app.schemas.diagnosis import QueryAnalyzeRequest

@router.get("/query-errors", response_model=ApiResponse)
def list_query_errors(level: str = "", conversation_id: str = "", limit: int = 50,
                      db: Session = Depends(get_db), role: str = Depends(get_role)):
    denied = _manager_only(role)
    if denied: return denied
    q = db.query(QueryTrace)
    if level: q = q.filter(QueryTrace.final_level == level)
    if conversation_id: q = q.filter(QueryTrace.conversation_id == conversation_id)
    rows = q.order_by(QueryTrace.id.desc()).limit(limit).all()
    return ApiResponse(data=[{"trace_id": r.id, "conversation_id": r.conversation_id,
                              "question": r.question, "response_type": r.response_type,
                              "final_level": r.final_level, "final_code": r.final_code,
                              "created_at": r.created_at} for r in rows])


@router.get("/query-traces/{trace_id}", response_model=ApiResponse)
def get_query_trace(trace_id: int, db: Session = Depends(get_db), role: str = Depends(get_role)):
    denied = _manager_only(role)
    if denied: return denied
    ev = agent.collect_query_evidence(db, trace_id)
    if not ev: return ApiResponse(code="NOT_FOUND", message="轨迹不存在")
    return ApiResponse(data=ev)


@router.post("/analyze-query", response_model=ApiResponse)
def analyze_query(req: QueryAnalyzeRequest, db: Session = Depends(get_db), role: str = Depends(get_role)):
    denied = _manager_only(role)
    if denied: return denied
    ev = agent.collect_query_evidence(db, req.trace_id)
    if not ev: return ApiResponse(code="NOT_FOUND", message="轨迹不存在")
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    if not setting or not setting.value:
        return ApiResponse(code="API_KEY_MISSING", message="请先在个人中心配置 DeepSeek API Key")
    result = agent.analyze(ev, DeepSeekClient(api_key=setting.value))
    result["evidence"] = ev
    if result.get("llm_error"):
        return ApiResponse(code="LLM_ERROR", message=result["llm_error"], data=result)
    return ApiResponse(data=result)
```
`chat_diagnosis` 分流：当 `req.subject_type == "query"` 时用 `req.trace_id` 走 `collect_query_evidence`，
并以 `subject_id=req.trace_id, subject_type="query"` 调 `diag_ctx.ask`；task 路径保持不变。

- [ ] **Step 5: 运行验证通过**

Run: `py -m pytest tests/test_diagnosis_query.py tests/test_diagnosis_api.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add backend/app/schemas/diagnosis.py backend/app/routers/diagnosis.py backend/tests/test_diagnosis_query.py
git commit -m "feat: 诊断路由新增检索错误列表/轨迹/分析端点"
```

---

## Task 10: 前端诊断页新增标签

**Files:**
- Modify: `frontend/src/api/diagnosis.js`
- Modify: `frontend/src/pages/Diagnosis/index.jsx`

- [ ] **Step 1: api 封装**

`frontend/src/api/diagnosis.js` 追加：
```javascript
export function getQueryErrors(level = '', conversationId = '') {
  const p = new URLSearchParams()
  if (level) p.set('level', level)
  if (conversationId) p.set('conversation_id', conversationId)
  return get(`/api/diagnosis/query-errors?${p.toString()}`)
}
export function getQueryTrace(traceId) { return get(`/api/diagnosis/query-traces/${traceId}`) }
export function analyzeQuery(traceId, errorLog = '') {
  return post('/api/diagnosis/analyze-query', { trace_id: traceId, error_log: errorLog })
}
```
`chatDiagnosis` 增参：`export function chatDiagnosis(taskId, question, sessionId = null, errorLog = '', subjectType = 'task', traceId = null) { return post('/api/diagnosis/chat', { task_id: taskId, question, session_id: sessionId, error_log: errorLog, subject_type: subjectType, trace_id: traceId }) }`

- [ ] **Step 2: 诊断页加标签与轨迹视图**

在 `pages/Diagnosis/index.jsx` 顶部加标签切换（`入库任务` / `检索问答`）。检索问答标签：
- 挂载调 `getQueryErrors(levelFilter)`，渲染每轮条目：问题摘要 + `final_level` 徽标（error 红 `#f85149`、degraded 黄 `#d29922`、normal 灰）+ `final_code` + 时间。
- 顶部 `<select>` 级别过滤（全部/error/degraded/normal）。
- 点击条目 → `getQueryTrace(traceId)` 展开多步轨迹（每步 step/level 颜色/code/message/metrics）。
- "分析"按钮 → `analyzeQuery(traceId)` 渲染 `analysis`（markdown，复用现有渲染），并接入现有 `chatDiagnosis(..., 'query', traceId)` 多轮追问。
- 沿用 `Loading`/`ErrorMessage`/`card`/`role-chip` 组件与样式。

- [ ] **Step 3: 构建验证**

Run: `cd d:\opencodeTest\RAG\frontend; npm run build`
Expected: 构建成功（`built in ...`）

- [ ] **Step 4: 提交**

```bash
git add frontend/src/api/diagnosis.js frontend/src/pages/Diagnosis/index.jsx
git commit -m "feat: 诊断页新增检索/问答错误标签与轨迹分析"
```

---

## Task 11: 全量回归

- [ ] **Step 1: 后端全量**

Run: `cd d:\opencodeTest\RAG\backend; py -m pytest tests/ -q`
Expected: 全部 passed（允许既有 skip）；重点关注 `test_agent_chain.py`、`test_diagnosis_*.py`、`test_query_trace.py`。

- [ ] **Step 2: 修复回归**

若 `_mock_model` 的结构化路由因新增字段受影响，按既有模式修 mock（不改产品代码语义）。

- [ ] **Step 3: 前端构建**

Run: `cd d:\opencodeTest\RAG\frontend; npm run build`
Expected: 成功。

- [ ] **Step 4: 提交（如有修复）**

```bash
git add -A
git commit -m "test: 检索错误诊断全量回归修复"
```

---

## Self-Review

**Spec 覆盖：** §3 分级→Task1/3；§4 架构→Task2/3/5；§5 trace+埋点→Task2/3；§6 模型→Task4；§7 收口→Task5/6；§8.1 采集分流→Task7；§8.2 上下文→Task8；§8.3 路由→Task9；§9 前端→Task10；§10 降级→各 Task 的 try/except；§11 测试→各 Task Step1 + Task11；§13 陷阱→Task2(覆盖语义)/Task3(纯数据)/Task1(默认False)/Task7(分流)/Task5(请求级db)。无遗漏。

**类型一致：** `_step(step, level, code, message, **metrics)`、`_classify_trace(trace)->(level,code)`、`_record_trace(db, conversation_id, question, resp, trace)`、`collect_query_evidence(db, trace_id)`、`QueryTrace` 字段、`subject_type/subject_id` 在各 Task 间引用一致。

**占位符：** 无 TBD/TODO，代码步均给出可落地内容。

> 备注：Task3/8 对既有节点函数体给出了埋点后的完整返回；实现时以现有 `agent_chain.py`/`context.py` 实际代码为基线，仅追加 trace/subject 相关键，勿改动既有检索/权限语义。
