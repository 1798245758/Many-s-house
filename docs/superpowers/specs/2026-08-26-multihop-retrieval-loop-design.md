# 多跳检索循环设计（依赖驱动，单次请求内迭代检索）

> 日期：2026-08-26　状态：已确认（用户确认依赖链式多跳 + 累积统一重排，
> 循环参数由设计方定义）

## 1. 背景与目标

典型多跳问题存在实体依赖：Q「张三公司的总部在哪个国家?」需先检索
「张三的公司」拿到公司名，再检索「该公司总部」才能作答。当前链路
单次检索无法覆盖此类依赖链。本设计在 **一次检索请求内** 迭代多轮检索，
后一跳查询由 LLM 基于已检索证据推理生成，循环直到问题解决。

## 2. 关键决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| 循环驱动 | 依赖链式：每跳后由 LLM 基于证据决定下一跳查询或收敛 |
| 候选聚合 | 每跳按 chunk_id 去重累积，循环结束后统一余弦重排 |
| 循环参数 | 由设计方定义，置于 `config.py`，环境变量可覆盖 |
| 触发条件 | 仅 search 分支；rewrite 无子问题时队列为空，首跳后直接重排，行为与现状一致（零额外调用） |

## 3. 图结构改造（`agent_chain.py` search 分支）

```
intent →(search)→ search_plan → embed → search ─┐
                                ↑                 ↓
                                │            route_search
                                │      ┌─ rerank（已终止/队列空/达上限）→ context → answer
                                │      └─ hop（队列非空 且 hop < HOP_MAX_HOPS）
                                │                 ↓
                                └── hop_reason（LLM 依赖推理）→ route_after_hop
                                          ├─ rerank（finished）
                                          └─ embed（next，检索本跳新查询）
```

- **`search_plan` 节点**：初始化循环。`current_query` = 现有合并关键词串
  （首跳）；`query_queue` = rewrite 的 `sub_questions`；`hop_count=0`、
  `all_hits=[]`。
- **`embed` / `search` 节点**：对 `current_query` 向量化并混合检索
  （`top_k=HOP_TOP_K_PER_HOP`，员工角色仍排除经理专属文档）；命中按
  `chunk_id` 去重累积进 `all_hits`，`hop_count += 1`。
- **`route_search` 条件边**：`loop_finished` / 队列耗尽 / `hop_count ≥
  HOP_MAX_HOPS` 任一成立 → `rerank`；否则 → `hop_reason`。
- **`hop_reason` 节点**：调用 `hop_chain` 推理（见下节），消费队首：
  - `finished=true` → 置 `loop_finished=True`，`route_after_hop` 直接进重排；
  - `finished=false` → `next_keywords` 空格拼接（兜底 `next_query`）作为
    `current_query`，队列出队首，回到 `embed`（本跳新查询仍会检索）。
  - 注意：`hop_reason` 出口必须为条件边（实施中发现固定边会让终止判定后
    仍被送去检索；而用队列判空又会跳过最后一跳的查询串）。
- **`rerank` 节点**：对 `all_hits` 全量统一余弦重排，截断保留
  `HOP_TOP_K_CONTEXT` 个写入 `ranked`；后续 `route_hits` / `context` /
  `answer` 逻辑不变。

## 4. 新模块 `hop_chain.py`

路径：`backend/app/services/retrieval/hop_chain.py`。

```python
class HopDecision(BaseModel):
    finished: bool          # 证据是否足以回答主问题
    next_query: str = ""    # 下一跳完整检索问句（自包含）
    next_keywords: list[str] = []  # 下一跳关键词（2~6 个）
```

- `HOP_PROMPT` 输入：主问题（改写后）+ 已检索证据（累积候选 Top3 内容
  截断摘要）+ 剩余子问题列表；要求基于证据中已拿到的事实（如实体名）
  生成自包含的下一跳查询。
- `build_hop_reasoner(api_key, model=None)`：返回
  `reasoner(question, evidence, remaining) -> HopDecision`；
  `method="function_calling"`（DeepSeek 约束）。
- **降级策略**（模块内完成）：异常时若 `remaining` 非空返回
  `finished=False, next_query=remaining[0]`（顺序消费子问题），
  否则 `finished=True`；保证可用性优先。

## 5. 循环参数（`config.py`，环境变量可覆盖）

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `HOP_MAX_HOPS` | 3 | 总检索跳数上限（首跳 + 最多 2 跳） |
| `HOP_TOP_K_PER_HOP` | 5 | 每跳混合检索候选数 |
| `HOP_TOP_K_CONTEXT` | 6 | 统一重排后入上下文的 chunk 数 |

收敛条件（任一满足即退出循环）：`HopDecision.finished=true`（本跳新查询不再生成）、
子问题队列耗尽、跳数达 `HOP_MAX_HOPS`。

## 6. 成本与兼容性

- 无子问题的普通问题：队列为空 → 首跳后直接重排，零额外调用，
  与现状行为一致；
- 最坏情况（3 跳）：3 次混合检索 + 3 次本地 BGE 向量 + 最多 2 次推理
  LLM 调用；
- 门禁、意图关卡、`QueryResponse` 结构、前端均不变。

## 7. 测试计划

- 新建 `tests/test_hop_chain.py`：finished 判定、next 推进、
  异常降级消费队首 / 队列空时终止；
- 更新 `tests/test_agent_chain.py`：
  - mock `hybrid_search` 与 `vector_store.get_embeddings`，验证多跳
    累积去重与 `finished` 终止；
  - 推理持续 next 时 `hybrid_search` 恰被调用 `HOP_MAX_HOPS` 次；
  - 推理链异常时顺序消费子问题队列；
  - 现有单跳用例（无子问题）不回归；
- 回归：`py -m pytest tests/ -v`（`USE_BGE_MODEL=false`，`backend` 目录）。
