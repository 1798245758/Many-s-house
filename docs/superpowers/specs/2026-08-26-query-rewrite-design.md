# Query 理解阶段设计（改写 + 分解 + 关键词优化）

> 日期：2026-08-26　状态：已确认（用户批准全部推荐选项后授权自主实施）

## 1. 背景与目标

当前在线链路为：

```
START → permission_gate → intent（classify + slot 子图）→ 分流 → 检索/直答 → END
```

意图分类与槽位提取直接基于用户原始问题，口语化、指代省略、复合问句会拉低
分类准确率与检索关键词质量。本设计在 **intent 之前** 增加 Query 理解阶段，
一次 LLM 调用完成：

- **改写**：口语转书面、补全省略成分，产出规范问题；
- **分解**：复合问题拆为 0~3 个子问题（简单问题为空列表）；
- **关键词优化**：产出 2~6 个覆盖原问题与全部子问题的检索关键词。

## 2. 关键决策（用户已确认）

| 决策点 | 选择 |
| --- | --- |
| LLM 调用方式 | 单次结构化输出（一次调用同时产出三项，延迟最低） |
| 子问题用法 | 关键词合并进 search_query，仍走单次混合检索（不做 Multi-Query） |
| 改写结果用途 | 意图识别、direct 直答、最终回答全部使用改写后问题 |
| 代码位置 | 主图 permission_gate 与 intent 之间新增 `rewrite` 节点 |

## 3. 新模块 `rewrite_chain.py`

路径：`backend/app/services/retrieval/rewrite_chain.py`，与 `intent_chain.py`
同目录同风格。

```python
class RewriteResult(BaseModel):
    rewritten_question: str   # 改写后的规范问题
    sub_questions: list[str]  # 子问题（0~3 个，简单问题为空）
    keywords: list[str]       # 优化后检索关键词（2~6 个，覆盖原问题与子问题）
```

- `REWRITE_PROMPT`：单模板，要求三项输出；明确"不改变用户原意、不回答问题本身"。
- `build_query_rewriter(api_key, model=None)`：返回
  `rewriter(question) -> RewriteResult`；`with_structured_output` 必须显式
  `method="function_calling"`（DeepSeek 不支持 json_schema response_format）。
- **降级策略**：任一异常返回
  `RewriteResult(rewritten_question=原问题, sub_questions=[], keywords=[原问题])`，
  记录 warning 日志，保证可用性优先。

## 4. 主图改造 `agent_chain.py`

```
START → permission_gate（仍用原问题，确定性门禁不变）
          └─ 放行 → rewrite（新增）→ intent → 分流 ...
```

- `RAGState` 新增字段：`rewritten_question` / `sub_questions` /
  `rewrite_keywords`。
- **rewrite 节点**：惰性构建 rewriter（闭包持有 api_key 与 model），
  门禁短路路径零初始化成本。
- **intent 节点**：`gate(state["rewritten_question"])`，分类与槽位提取
  基于改写后问题。
- **search_query 合成**（在 intent 节点）：
  `rewrite_keywords + slot.search_keywords` 合并去重后空格拼接；
  两者皆空时兜底 `state["question"]`。子问题关键词已在改写阶段并入
  `keywords`，因此仍是单次混合检索。
- **direct / answer 节点**：使用 `rewritten_question` 作答；
  原始问题保留在 `state["question"]`，用于门禁判断、历史记录与前端展示。
- 延迟影响：每个非短路请求 +1 次 LLM 调用。

## 5. 明确不做（YAGNI）

- 不改 API 响应结构（`QueryResponse` 不变），改写结果仅记日志；
- 不做 Multi-Query 多次检索与结果合并；
- 不改前端。

## 6. 测试计划

- 新建 `tests/test_rewrite_chain.py`：
  - 正常返回（mock `with_structured_output` 返回链，`.return_value` 语义）；
  - LLM 异常降级为原问题。
- 更新 `tests/test_agent_chain.py`：
  - `with_structured_output.side_effect` 顺序改为 **rewrite → classify → slot**；
  - 验证 search 分支 search_query 为关键词合并结果（patch `hybrid_search`
    断言入参）；
  - 验证 direct 分支用改写后问题；
  - 门禁短路用例保持 `with_structured_output.assert_not_called()`。
- 回归：`py -m pytest tests/ -v`（`USE_BGE_MODEL=false`，从 `backend` 目录执行）。
