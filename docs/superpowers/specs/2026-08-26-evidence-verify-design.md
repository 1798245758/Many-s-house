# 证据校验节点设计（verify：四条件校验 + 三种硬停）

> 日期：2026-08-26　状态：已确认并按实施修订（route_verify 改用 `verify_retrying` 显式标志）

## 1. 背景与目标

多跳检索循环当前以 `hop_reason` 的 `finished` 判定收尾，缺少对证据质量
的最终把关。本设计在 `rerank` 之后增加 **证据校验节点**：一次结构化
LLM 调用判断证据是否"充分且一致"，不满足则继续循环补充检索；特定条件
下强制停止，但**不做确定性回答**。

## 2. 关键决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| 补充策略 | 针对同一问题生成**不同角度**的查询（同义词替换、上下位概念、换资料类型视角）+ 补充关键词，与 hop_reason 风格一致，不消费子问题队列 |
| 硬停表达 | `QueryResponse` 新增 `evidence_status` 字段 + answer prompt 对冲指令 |
| 校验位置 | `rerank` 之后、`context` 之前；仅 search 分支生效 |
| 参数 | 由设计方定义，置于 `config.py` |

## 3. 图结构改造

```
rerank → route_hits ─┬─ 无命中 → END（固定话术，不变）
                     └─ 有命中 → verify（单次结构化 LLM 调用）
                                   ├─ sufficient           → context → answer（verified）
                                   ├─ 硬停（三条件之一）    → context → answer（uncertain，对冲）
                                   └─ 可继续               → embed → search → rerank → verify …
```

- **`verify_node`**：惰性构建校验链；输入主问题（改写后）+ 子问题清单
  （关键问题映射）+ 精排后带来源的证据。四条件全过 →
  `evidence_status="verified"`；不过则计算硬停条件：
  - 未硬停：写入 `current_query`（补充查询，`next_keywords` 空格拼接，
    兜底 `next_query`）→ 条件边回 `embed`；
  - 硬停：`evidence_status="uncertain"` + 记录 `missing_aspects` → `context`。
- **`route_verify` 条件边**：以 `verify_node` 本轮写入的显式标志
  `verify_retrying` 为准——`True` → `embed` 补检，`False`（verified/
  uncertain 均置 `False`）→ `context`。**实施修正**：不能用
  "`evidence_status` 是否已写"判断——同一图执行内状态会残留，
  二次校验时上一轮的 `verified/uncertain` 残留值会把补检循环
  误判为定局，直接跳过后续校验。
- **硬停判定**（在 `verify_node` 内，任一成立即停）：
  1. `hop_count >= HOP_MAX_HOPS`（补充检索同样计入轮数）；
  2. 连续 `VERIFY_MISSING_STREAK=2` 轮 `missing_aspects` 完全相同
     （`last_missing`/`missing_streak` 状态跟踪）；
  3. `no_new_hits=True`（上一轮检索零新增候选，`search_node` 记录）。
- **`search_node`**：新增返回 `no_new_hits`（去重后无新增）。
- **`answer_node`**：`evidence_status=="uncertain"` 时向 prompt 追加对冲
  指令——证据不足、须用不确定语气、说明缺失方面（注入
  `missing_aspects`）、不得给出确定性结论。

## 4. 新模块 `verify_chain.py`

路径：`backend/app/services/retrieval/verify_chain.py`。

```python
class VerifyResult(BaseModel):
    sufficient: bool            # 四条件全部满足
    missing_aspects: list[str]  # 缺失方面（无证据的关键问题）
    next_query: str = ""        # 同一问题换角度的补充查询
    next_keywords: list[str] = []
```

- `VERIFY_PROMPT` 四条件：①关键问题（主问题+各子问题）均有证据覆盖；
  ②来源满足权威性（正式制度/手册/培训资料为权威，来源不明存疑）；
  ③引用能支持结论且相互不矛盾；④未发现权限和安全问题。
  不满足时输出缺失方面与换角度补充查询。
- `build_evidence_verifier(api_key, model=None)`：返回
  `verifier(question, sub_questions, evidence) -> VerifyResult`；
  `method="function_calling"`。
- **异常降级**：视为 `sufficient=True` 放行（避免校验故障把循环卡死，
  可用性优先），记录 warning 日志。

## 5. Schema 与参数

- `QueryResponse` 新增 `evidence_status: str | None`（`verified` /
  `uncertain`，仅 search 分支有值，其余分支为 None）；
- `config.py` 新增 `VERIFY_MISSING_STREAK`（默认 2，环境变量可覆盖）；
  轮数上限复用 `HOP_MAX_HOPS`。

## 6. 成本与兼容性

- 每个 search 请求每轮 +1 次校验调用（最坏 3 轮 +3 次）；
- direct / refuse / clarify / 门禁逻辑不变；前端本次不改（字段已备好）。

## 7. 测试计划

- 新建 `tests/test_verify_chain.py`：四条件通过 / 不通过给出缺失方面与
  补充查询 / 异常降级放行；
- 更新 `tests/test_agent_chain.py`：
  - 校验通过 → `evidence_status="verified"`；
  - 校验不通过 → 补检一轮后通过（`hybrid_search` 恰两次）；
  - 缺失方面连续相同 → 硬停 `uncertain` 且回答 prompt 含对冲指令；
  - 达 `HOP_MAX_HOPS` 后校验不通过 → 硬停 `uncertain`；
  - 补充检索零新增候选 → 硬停 `uncertain`；
- 回归：`py -m pytest tests/ -v`（`USE_BGE_MODEL=false`，`backend` 目录）。
