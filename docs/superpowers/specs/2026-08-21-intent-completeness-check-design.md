# 意图识别与信息完整性检查设计

日期：2026-08-21
状态：已与用户确认

## 1. 目标

在在线问答链路最前端增加意图理解关卡：根据用户输入识别**任务、实体、时间、风险**，判断信息是否完整：

- 信息不完整 → 返回澄清问题，等待用户下一轮补充
- 命中安全风险（敏感/越界内容）→ 礼貌拒绝
- 完整且无需外部知识 → 直接回答
- 完整且需要事实性资料 → 进入现有检索管道

## 2. 关键决策（已与用户确认）

| 决策点 | 结论 |
|---|---|
| 澄清机制 | 无状态：后端返回澄清提示，用户下一轮重新提交补充信息 |
| "风险"定义 | 问题本身的安全风险（敏感/越界内容），非业务风险提醒 |
| 与 decision_chain 关系 | 新的意图链**替代**现有 decision_chain（检索关键词职责并入槽位提取） |
| 分析结果展示 | 前后端都展示（响应携带 IntentInfo，前端渲染标签） |
| 完整性判断严格度 | 宽松：仅当意图模糊到无法判断、或缺失要素直接导致歧义/检索失败时才澄清 |
| 实现方案 | 方案B：两阶段 LLM 调用（分类 + 槽位提取） |

## 3. 后端架构

### 3.1 新模块 `backend/app/services/retrieval/intent_chain.py`

不引入 LangGraph，两条 LCEL 链 + 普通函数编排。

**① classify_chain（第一阶段·分类）**

用 `model.with_structured_output()` 结构化输出：

- `action`: `refuse` | `clarify` | `direct` | `search`
  - `refuse`：问题含安全风险（违法、涉密、有害请求等）
  - `clarify`：信息不完整（宽松标准）
  - `direct`：闲聊/常识等无需知识库即可回答
  - `search`：涉及企业管理事实，需检索
- `risk_reason`: 命中 refuse 时的简要理由（其余场景为空串）

Prompt 内置宽松判断标准与四类 action 的判定示例。

**② slot_chain（第二阶段·槽位提取）**

仅在 `action != refuse` 时调用，结构化输出 `IntentInfo`：

- `task`: 任务类型（如"制度咨询""流程咨询""闲聊"）
- `entities`: 实体列表（如 ["胖东来", "员工休假"]）
- `time`: 时间信息，无则为 None
- `risk_note`: 可选风险备注，通常为 None
- `search_keywords`: 仅 action=search 时输出，2~6 个检索关键词（承接原 decision_chain 职责）
- `clarification_question`: 仅 action=clarify 时输出，面向用户的澄清问题

### 3.2 路由编排

入口函数（如 `intent_gate`）串联两阶段并分流：

```
用户问题 → classify_chain
  ├─ refuse  → 返回固定礼貌拒绝话术（不额外调用 LLM，不进入后续链路）
  ├─ clarify → 返回澄清问题 + 部分意图分析，本轮结束
  ├─ direct  → 复用现有 answer_chain（空上下文）生成直答
  └─ search  → slot_chain 提取关键词 → 现有后 5 段管道
               （向量化→检索→精排→上下文→回答）
```

### 3.3 `agent_chain.py` 改造

- 删除 `DECISION_PROMPT`、`decision_chain` 及 `bind_tools`/`@tool` 包装逻辑；检索改为直接调用 `hybrid_search`（关键词来自 slot_chain 的 `search_keywords`）
- `rag_chain_query` 入口先过意图关卡，按 action 分流；仅 `search` 分支进入向量化→检索→精排→上下文→回答管道
- 检索工具向量基准问题同步处理：精排使用 search_keywords 向量化结果（与现状 exec_tool 行为一致）

## 4. API 响应结构（`backend/app/schemas/query.py`）

```python
class IntentInfo(BaseModel):
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

| 场景 | response_type | answer | intent | clarification_question |
|---|---|---|---|---|
| 检索回答 | answer | 最终回答 | ✅ | 无 |
| 直答 | answer | 直答内容 | ✅ | 无 |
| 信息不完整 | clarification | 澄清引导语 | ✅（部分槽位） | ✅ |
| 安全风险 | refusal | 礼貌拒绝说明 | 可为 None | 无 |

- `QueryRequest` 不变（仍仅 `question` 字段）
- 历史记录照常落库，`answer_text` 存澄清问题/拒绝说明

## 5. 前端改造

`frontend/src/pages/Chat/index.jsx` + `api/query.js`：

1. **意图分析标签区**：回答卡片顶部以标签展示 任务类型（蓝色）、实体（灰色逐个）、时间（有值才显示）、风险备注（橙色警示，有值才显示）
2. **澄清交互**：`response_type === "clarification"` 时卡片改为琥珀色提示样式，显示澄清问题，附"补充后重新提问"按钮（将澄清问题原文填入输入框，用户修改后提交）
3. **拒绝样式**：`response_type === "refusal"` 时红色提示样式展示拒绝说明
4. `submitQuery` 透传新字段，调用方式不变

## 6. 错误处理

- 意图链 LLM 调用失败/结构化解析失败 → 降级：action 视为 search，用原始问题作为检索关键词，走完整检索管道（等价于现状的"总是检索"行为，保证可用性优先）
- slot_chain 失败但 classify 成功 → search 分支用原问题作为检索关键词兜底

## 7. 测试计划

- 单元测试：mock LLM 输出，覆盖四种 action 分流与降级路径（遵循 `USE_BGE_MODEL=false` 与向量隔离 fixture 约定）
- 手动验证：典型用例——
  - "你好" → direct 直答
  - "胖东来的员工休假制度" → search 检索回答
  - "说说那个制度" → clarify 澄清
  - 敏感/违法类问题 → refusal
- 前端验证：三种 response_type 的展示与澄清回填交互
