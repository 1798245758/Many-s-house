# RAG 检索错误分析与统一诊断 Agent 设计

> 日期：2026-09-28　状态：待审阅
> 目标：为 RAG 检索/问答链路增加"每轮多步轨迹日志 + 三级错误状态机 + 点击日志触发错误分析"，
> 并与现有"入库任务错误诊断 Agent"统一为一个诊断 Agent（共用入口与多轮上下文，按错误类型分流）。

## 1. 背景与目标

现有诊断 Agent（`app/services/diagnosis/`）只覆盖**异步文档入库任务**的失败：读 SqliteSaver
里的任务状态机 checkpoint + 文档信息 + 错误日志，交 LLM 分析根因，经理专用，带 L0–L3 多轮上下文。

而 **RAG 检索/问答链路**（`agent_chain.py` 的 LangGraph，短期记忆用 MemorySaver）目前每轮只把
"问题/答案/来源"写进 `QueryHistory`，**不持久化任何逐步状态流转**。用户需要：

1. 检索过程维护的多个状态机（意图子图、多跳循环、证据校验循环）的**每轮多步轨迹**可见；
2. 轨迹中的**错误/降级步骤用特殊颜色高亮**；
3. **点击某轮/某步日志 → 错误分析 Agent 输出可能根因**；
4. 错误状态机覆盖常见 RAG 错误（如**候选集为空**等）。

## 2. 关键决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| 与现有诊断 Agent 关系 | **统一为一个诊断 Agent**：同时覆盖入库任务失败 + RAG 检索错误，共用入口与上下文，按错误类型分流 |
| 日志粒度 | **每轮展开为多步轨迹**（intent→rewrite→search→rerank→verify→answer 每步一条），出错步骤高亮 |
| 错误分级 | **三级**：正常(灰)/降级警告(黄)/错误(红)；权限拦截、安全拒绝、澄清等确定性短路算**正常** |
| 前端入口 | 复用现有**诊断页（经理专用）**，新增"检索/问答错误"标签 |
| 埋点与持久化 | **方案 B**：图内 `trace` 通道累积 + 收口一次性分级落专用表 `QueryTrace`；短期记忆 MemorySaver **不改动** |

## 3. 错误状态机分级清单

轮级严重度 = 各步最高级（error > degraded > normal）。

### 🔴 错误（检索彻底失败/异常）
| 编号 | 状态 | 触发点 |
| --- | --- | --- |
| E1 | 候选集为空 | rerank 后 `ranked==0`（无命中，走 NO_RESULT） |
| E2 | 向量化异常 | `embed_query` 抛错 |
| E3 | 检索执行异常 | `hybrid_search` 抛错 |
| E4 | 回答生成彻底失败 | answer/direct 的 LLM 抛错且无兜底 |
| E5 | 整轮图执行异常 | `invoke` 未捕获异常（query.py 回退旧流水线/LLM_ERROR） |

### 🟡 降级警告（有结果但质量受损或走了兜底）
| 编号 | 状态 | 触发点 |
| --- | --- | --- |
| W1 | 证据不足硬停 | `evidence_status=uncertain`（附缺失方面） |
| W2 | 改写降级 | rewrite LLM 失败 → 原问题兜底 |
| W3 | 意图识别降级 | intent LLM 失败 → 默认走检索 |
| W4 | 多跳推理降级 | hop LLM 失败 → 顺序消费子问题队列 |
| W5 | 证据校验降级 | verify LLM 失败 → 视为充分放行 |
| W6 | 达跳数上限仍未充分 | `hop_count>=HOP_MAX_HOPS` |

### ⚪ 正常（含确定性短路）
| 编号 | 状态 |
| --- | --- |
| N1 | 正常回答（verified 且命中） |
| N2 | 权限拦截（permission_denied） |
| N3 | 安全拒绝（refuse） |
| N4 | 澄清（clarify） |
| N5 | 直答闲聊（direct） |

> E5 由 `query.py` 的 except 分支记录（整轮 invoke 失败时，构造一条 error 级 QueryTrace）。

## 4. 架构总览

```
每轮问答（rag_chain_query）：
  invoke 前：入口节点 permission_gate 置 trace=[]（每轮重置）
  图执行：各节点把"本步记录"append 进 state["trace"]（覆盖式累积）
          各链兜底分支回传 degraded=True → 节点据此打 W2–W5
  invoke 后（收口，所有返回分支）：
    _remember_turn（短期记忆，已有）
    _extract_memories（长期记忆，已有）
    _record_trace（新增）：由 final["trace"] 计算 final_level=max，落一条 QueryTrace（try/except 不阻断）

诊断（经理专用，统一 Agent）：
  GET  /api/diagnosis/query-errors            列出 QueryTrace（按 level/conversation 过滤）
  GET  /api/diagnosis/query-traces/{trace_id} 单轮多步轨迹（不调 LLM）
  POST /api/diagnosis/analyze-query           采集问答证据 → RETRIEVAL_SYSTEM_PROMPT → LLM 根因
  POST /api/diagnosis/chat（复用）            多轮追问，subject=(query, trace_id) 钉住问答证据
```

## 5. trace 数据结构与埋点（`agent_chain.py`）

- `RAGState` 新增普通字段 `trace: list`（**非 Annotated reducer**，覆盖语义；因图内节点顺序执行，
  每节点返回 `state.get("trace",[]) + [record]` 即累积，入口节点返回 `[]` 即重置）。
- 步记录为纯数据 dict（可 msgpack 序列化，与 all_hits 同约束）：
  ```python
  {"step": "rerank", "level": "error", "code": "E1",
   "message": "候选集为空", "metrics": {"ranked": 0}, "ts": "2026-09-28T10:00:00"}
  ```
- 辅助函数：
  ```python
  def _step(step, level, code="", message="", **metrics) -> dict: ...
  ```
- 埋点点位与产出：
  - `permission_gate`：重置 `trace=[]`；命中门禁时 append N2。
  - `intent`：append action（refuse→N3 / clarify→N4 / direct→N5 / search→N1 占位）；`degraded` → W3。
  - `rewrite`：`degraded` → W2。
  - `embed`：异常 → E2（try/except 内记录后按现有降级/抛出策略处理）。
  - `search`：记录 `candidate_count`；异常 → E3。
  - `hop_reason`：`degraded` → W4。
  - `rerank`：`ranked==0` → E1。
  - `verify`：`uncertain` → W1（附 missing_aspects）；`degraded` → W5；`hop_count>=HOP_MAX_HOPS` → W6。
  - `answer`/`direct`：正常 append N1；LLM 抛错 → E4。
- **降级标记回传**（唯一需改各链内部）：给 `RewriteResult`、intent gate 返回、`HopDecision`、
  `VerifyResult` 各加 `degraded: bool = False`，在其 except 兜底分支置 True。默认 False，向后兼容。

## 6. 数据模型（新增 `app/models/query_trace.py`）

```python
class QueryTrace(Base):
    __tablename__ = "query_trace"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(String, index=True)      # 会话（thread_id），未传兜底 "default"
    question = Column(Text, nullable=False)
    answer_snippet = Column(Text)                      # 截断 _ANSWER_SNIPPET_LEN
    response_type = Column(String)                     # answer|clarification|refusal|permission_denied
    final_level = Column(String, index=True)           # normal|degraded|error
    final_code = Column(String)                        # 触发 final_level 的最高级 code
    steps = Column(Text)                               # JSON：多步轨迹数组
    created_at = Column(String, server_default=func.datetime('now'))
```

- 与 `QueryHistory` 分离：QueryHistory 面向用户历史页（无 conversation_id）；QueryTrace 面向诊断页。
- `init_db` 的轻量迁移需确保新表建出（Base.metadata.create_all 覆盖）。

## 7. 收口分级与落库（`agent_chain.py`）

```python
_LEVEL_ORDER = {"normal": 0, "degraded": 1, "error": 2}

def _classify_trace(trace: list) -> tuple[str, str]:
    """返回 (final_level, final_code)：取各步最高级；同级取最后一个"""

def _record_trace(db, conversation_id, question, resp, trace):
    """落一条 QueryTrace；db 为 None（部分测试/短路）或写库异常时静默跳过，不阻断问答"""
```

- 在 `rag_chain_query` 所有返回分支（permission_denied/refuse/clarify/no-result/answer）
  于 `_remember_turn`、`_extract_memories` 之后调用 `_record_trace`。
- `query.py` 的整轮 invoke 失败 except 分支：构造并落一条 `final_level="error", final_code="E5"` 的 QueryTrace。

## 8. 统一诊断 Agent 扩展

### 8.1 证据采集（`diagnosis/agent.py`）
- 新增 `collect_query_evidence(db, trace_id) -> dict | None`：读 QueryTrace 一行，组织为
  `{trace_id, conversation_id, question, answer_snippet, final_level, final_code, response_type, steps}`。
- 新增 `_query_evidence_to_text(evidence)`：把多步轨迹按时间序文本化（step/level/code/message/metrics）。
- 新增 `RETRIEVAL_SYSTEM_PROMPT`：定位"RAG 检索链路诊断专家"，输出三段（根因分析/关键证据/修复建议），
  只依据轨迹证据推理。
- `analyze` 泛化：按 evidence 是否含 `steps`（问答）或 `events`（任务）选择对应 prompt 与文本化函数。

### 8.2 多轮上下文（`diagnosis/context.py`）
- 钉住主体从 `task_id` 泛化为 `(subject_type, subject_id)`：`("task", task_id)` 或 `("query", trace_id)`。
- `ContextManager` 与 `ask()` 增加 `subject_type` 参数；L0–L3、滚动摘要、LRU、压缩机制**完全复用**。
- 换主体（不同 trace/task）时重建会话，避免上下文污染（沿用现有 task_id 变更即重建的逻辑）。

### 8.3 路由（`routers/diagnosis.py`，均经理专用，复用 `_manager_only`）
- `GET /api/diagnosis/query-errors?level=&conversation_id=&limit=50` → QueryTrace 列表（倒序）。
- `GET /api/diagnosis/query-traces/{trace_id}` → 单轮多步轨迹（不调 LLM）。
- `POST /api/diagnosis/analyze-query {trace_id, error_log?}` → 采集问答证据 + LLM 分析。
- `POST /api/diagnosis/chat`：请求体增加 `subject_type`（默认 "task" 向后兼容）与 `trace_id`；
  query 类走 `collect_query_evidence`。
- `schemas/diagnosis.py` 增加 `QueryAnalyzeRequest`，`DiagnosisChatRequest` 增 `subject_type/trace_id`。

## 9. 前端

- `api/diagnosis.js` 新增 `getQueryErrors(level, conversationId)` / `getQueryTrace(traceId)` /
  `analyzeQuery(traceId, errorLog)`；`chatDiagnosis` 增加 subject 参数。
- 诊断页（`pages/Diagnosis`）新增标签「检索/问答错误」：
  - 左列：每轮条目（问题摘要 + `final_level` 徽标[红/黄/灰] + 时间），顶部按级别过滤下拉。
  - 展开：该轮多步轨迹（step 名 + level 颜色 + code + message + metrics），错误步红、降级步黄。
  - 点击某步或整轮"分析"按钮 → 右侧/弹层复用现有分析与多轮追问 UI（传 subject_type="query"）。
- 沿用 `ErrorMessage`/`Loading`/`card`/`role-chip` 等既有组件与样式。

## 10. 错误处理与降级

| 场景 | 行为 |
| --- | --- |
| trace 埋点内异常 | 单步记录失败不影响该节点主逻辑（埋点包裹 try/except） |
| QueryTrace 落库失败 / db 为 None | 静默跳过，问答正常返回 |
| 各链 LLM 失败 | 现有兜底不变，额外回传 degraded 标记 → 记 W 级 |
| collect_query_evidence 找不到 trace | 返回 NOT_FOUND |
| 分析 LLM 失败 | 复用现有 LLM_ERROR 降级（返回证据摘要） |

原则：**诊断是旁路增强，任何环节失败都不得影响主问答链路。**

## 11. 测试计划（TDD）

- `test_query_trace.py`（埋点与分级）：
  - 候选空 → 出现 E1、final_level=error；verify uncertain → W1、degraded；rewrite 降级 → W2；
    权限/拒绝/澄清短路 → final_level=normal；多跳降级 → W4；达跳数上限 → W6。
  - `_classify_trace` 取各步最高级。
  - 每轮 trace 重置（同 conversation 两轮不串）。
  - 落库：各分支各落一条 QueryTrace；db=None 或写库异常时问答仍正常返回。
- 各链 `degraded` 回传单测（mock 链抛错 → 结果 degraded=True）。
- `test_diagnosis_query.py`（API）：query-errors 列表经理专用/过滤、trace 详情、analyze-query 分流、
  chat subject_type=query 复用上下文。
- 回归：`py -m pytest tests/`（`USE_BGE_MODEL=false`，backend 目录）；前端 `npm run build`。

## 12. 边界与成本

- 每轮 +1 次 QueryTrace 写库；无额外 LLM 调用（分析仅在经理点击时触发）。
- trace 仅存于图 state 与 QueryTrace.steps（JSON），不进短期记忆的跨轮累积（每轮重置）。
- 埋点对各链的侵入仅限"兜底分支置 degraded=True"，默认 False 向后兼容。
- QueryTrace 与 knowledge.db 同库不同表；可按需加清理策略（保留最近 N 天/条）。

## 13. 关键实现陷阱

1. `trace` 用普通字段（覆盖语义）而非 reducer，靠入口节点置 `[]` 实现每轮重置；若误用累积 reducer
   会跨轮残留（MemorySaver 恢复带回上轮 trace）。
2. 步记录必须纯数据 dict（含 metrics），ORM/异常对象不可进 checkpoint。
3. `degraded` 字段加默认值 False，避免破坏既有链测试的结构化输出 mock 映射。
4. 统一诊断 Agent 的 `analyze`/`context` 分流按证据结构判定，勿把两类证据字段混用。
5. QueryTrace 落库用请求级 db（rag_chain_query 签名已有），勿在单例图节点内闭包捕获 db。
