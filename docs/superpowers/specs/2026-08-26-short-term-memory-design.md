# 短期记忆系统设计（MemorySaver Checkpointer：近期对话 + 任务状态）

> 日期：2026-08-26　状态：已确认（会话标识=前端 conversation_id，历史仅注入 rewrite，
> 状态不落库存 Checkpointer、进程退出即清理，历史 3 轮，刷新页面保留会话）

## 1. 背景与目标

系统跨轮次无状态：追问无法承接上下文（"那工资呢？"），澄清后用户补充信息
看不到上轮语境。本设计引入**短期记忆**：以 QA 问答对为粒度记录近期对话
消息与上一次任务完成状态，进程退出自动清理（不引入数据库持久化）。

## 2. 关键决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| 会话标识 | 前端生成 `conversation_id`（UUID），存 sessionStorage——刷新页面保留会话，关标签页新开会话；当后端 `thread_id` 用 |
| 历史用途 | 仅注入 `rewrite` 节点：指代消解 + 澄清承接；意图/检索/回答基于补全后的问题 |
| 状态存储 | **不落库**：任务状态作为字符串存 LangGraph MemorySaver Checkpointer，程序退出即清理；`QueryHistory` 表不动 |
| 历史轮数 | 最近 3 轮（reducer 自动截断） |

## 3. 数据流

```
第 N 轮：
  invoke(输入仅 {question, role}, config={thread_id: conversation_id, db})
    → MemorySaver 恢复上轮 state（conversation_history + last_task_status）
    → rewrite 读历史/状态 → 指代消解、澄清承接 → 正常全链路
  结束后：
    rag_chain_query 统一 graph.update_state(as_node="answer")
      追加本轮"问/答"条目（reducer 截断保 3 条）+ 本轮任务状态字符串
```

## 4. 实现要点

### 4.1 状态字段（agent_chain.py RAGState）

- `conversation_history: Annotated[list[str], _keep_recent_history]`：
  每条 `"问：…\n答：…（答案截断300字）"`；reducer `(old+new)[-3:]`
- `last_task_status: str`：覆盖式字符串

### 4.2 任务状态字符串映射

| 分支/结果 | 状态字符串 |
| --- | --- |
| permission_denied | `权限不足，未回答` |
| refusal | `已拒绝（安全风险）` |
| clarification | `等待用户澄清：<澄清问题>` |
| answer + evidence_status=uncertain | `已回答但证据不足` |
| answer 且无命中固定话术 | `知识库未找到相关信息` |
| 其余 answer | `已完成回答` |

### 4.3 图改造

- `graph.compile(checkpointer=MemorySaver())`；单例缓存键改为
  `(api_key, id(model))`（测试注入 mock 时同 mock 复用，保证多轮共享同一
  MemorySaver）
- `rewrite_node` 把 `conversation_history` + `last_task_status` 传给改写链；
  `rewrite_chain.py` prompt 增加"最近对话记录/上轮任务状态"段，无历史时占位"（无）"
- `search_plan_node` 重置上轮残留的循环字段（`loop_finished`/
  `verify_retrying`/`missing_streak`/`last_missing`/`evidence_status`/
  `missing_aspects`）——checkpointer 恢复会带回上一轮终态，不重置会让
  第二轮多跳循环被跳过
- 记忆写入收口在 `rag_chain_query` 末尾 `update_state`（所有分支统一，
  节点零改动）
- **实施修正（状态序列化约束）**：挂 Checkpointer 后每步都序列化整个
  state（msgpack），ORM 对象/自定义类实例会导致写入时崩溃。因此
  `search_node` 检索到 Chunk 后立即转纯数据 dict（id/content/
  document_name），`all_hits`/`ranked` 全链按字段访问

### 4.4 API 与前端

- `QueryRequest` 增加可选 `conversation_id`；未传兜底 `"default"`
- 前端 `Chat` 页：`sessionStorage` 缓存 `crypto.randomUUID()`，
  `submitQuery` 携带；刷新保留、关标签页失效

## 5. 测试计划（TDD）

- `test_rewrite_chain.py`：带历史时渲染 prompt 含历史条目与状态；无历史时占位
- `test_agent_chain.py`（`_mock_model` 的 structured_output mock 改为按
  schema 动态路由，天然支持多轮）：
  - 同会话两轮：第二轮改写 prompt 含第一轮问答与任务状态
  - 不同会话隔离：互不串
  - 循环状态重置：第一轮置 `loop_finished` 后，第二轮多跳仍正常推进
    （`hybrid_search` 调用次数证明）
- 回归：`py -m pytest tests/`（`USE_BGE_MODEL=false`，backend 目录）

## 6. 边界与成本

- 每轮 +0 次 LLM 调用（记忆只在既有 rewrite 调用内消费）；
- 内存占用随进程存活期线性增长（每会话 ≤3 条短文本），单用户场景可忽略；
- 后端重启后同 `conversation_id` 进来：无 checkpoint 等于新会话，行为安全降级。
