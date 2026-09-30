# 异步任务框架 + CheckPoint 状态机 设计文档

日期：2026-08-29
状态：已确认（方案 A：LangGraph 状态图 + SqliteSaver）

## 1. 背景与目标

当前 `POST /api/documents/upload` 与 `PUT /api/documents/{id}` 在请求内同步跑完
"解析→清洗→切块→向量化→写库"全流程，大文档会长时间挂住请求，前端只能干等。

目标：

1. 建立**通用异步任务框架**：接口秒回，后台线程池执行，文档入库（`doc_ingest`）是第一个任务类型。
2. 任务生命周期以**六态状态机**建模，每次状态流转写入 **LangGraph Checkpoint**（SqliteSaver 落 SQLite），
   `get_state_history` 即完整流转事件日志，可追溯。
3. 用户在等待期间通过 **SSE 推流**实时看到"状态 + 阶段进度"。

## 2. 状态机定义

六态（结合文档入库业务调整语义）：

| 状态 | 语义 |
|---|---|
| `queued` | 已创建，等待线程池空闲工人（"创建"态） |
| `running` | 工人已接手，正在执行某阶段 |
| `blocked` | 外部依赖等待：向量化服务异常的退避重试窗口期，恢复后回 `running` |
| `cancelled` | 终态。排队任务立即取消；运行中任务在下一个阶段边界生效 |
| `failed` | 终态。任一阶段异常且重试耗尽 |
| `succeeded` | 终态。全部阶段完成 |

合法转移（唯一事实源，非法转移抛 `IllegalTaskTransition`）：

```
queued    -> running | cancelled
running   -> blocked | succeeded | failed | cancelled
blocked   -> running | failed | cancelled
```

阶段进度（`stage` + `progress` 固定权重）：

| 阶段 | extract | clean | chunk | embed | index |
|---|---|---|---|---|---|
| 进度区间 | 0→30% | 30→40% | 40→50% | 50→90% | 90→100% |

## 3. 架构

### 3.1 模块划分（新建 `backend/app/services/tasks/`）

| 文件 | 职责 |
|---|---|
| `machine.py` | 状态常量、`LEGAL_TRANSITIONS`、`transition()` 校验 |
| `state.py` | `TaskState` TypedDict（全纯数据，ORM 对象绝不进图）、初始状态工厂 |
| `cancel.py` | 进程内取消标志集合（`request_cancel` / `is_cancel_requested`） |
| `ingest_graph.py` | 入库任务状态图：`extract → clean → chunk → embed ⇄ wait_retry → index → finalize` |
| `runner.py` | SqliteSaver 单例、`ThreadPoolExecutor(max_workers=2)`、提交/快照/取消/重启清理 |

### 3.2 任务图

- 每个阶段节点：入口检查取消标志 → `announce`（经 `update_state` 提前写"进行中"checkpoint，
  让长阶段执行期间前端能看到"向量化中…"）→ 执行工作 → 返回完成态（阶段/进度/消息）。
- `embed` 节点：异常且重试未耗尽 → 返回 `blocked` + `retry_count+1`；条件边路由到
  `wait_retry`（退避 sleep 后回 `embed`）；重试耗尽 → `failed`。
  每轮显式返回状态，不依赖残留字段路由。
- `finalize` 节点（所有路径收口）：
  - `succeeded`：index 阶段 `save_chunks` 已置 `ready`，无额外动作；
  - `failed` → `Document.status='error'`；`cancelled` → `'cancelled'`；两者均清理已写 chunks 与 Chroma 向量。
- Checkpoint：编译挂 `SqliteSaver` 进程级单例（`knowledge.db`，`check_same_thread=False`，
  `isolation_level=None`），每个 super-step 自动落盘 = 每次流转写入 checkpoint。
- 创建即落盘：提交任务时 `update_state(config, initial, as_node="__start__")` 先写 `queued`
  checkpoint，工人线程 `invoke(None, config)` 从断点续跑。

### 3.3 API

| 端点 | 行为 |
|---|---|
| `POST /api/documents/upload` | 秒回。保存文件、建 Document 行（`status=processing`、`task_id`）、每文件提交一个任务，返回 `{tasks:[{filename, document_id, task_id}], errors:[...]}`。文件类型校验仍同步做 |
| `PUT /api/documents/{id}` | 秒回，返回 `{task_id}` |
| `GET /api/tasks/{task_id}` | 读最新 checkpoint（刷新/断线兜底），仅经理 |
| `GET /api/tasks/{task_id}/stream` | SSE：0.5s 轮询 `get_state`，变化推 `progress` 事件，终态推 `done` 后关流。鉴权走 `?role=` 查询参数（EventSource 无法设请求头），缺省按 employee |
| `POST /api/tasks/{task_id}/cancel` | 置取消标志；`queued` 任务同时直接把 checkpoint 改为 `cancelled` |

### 3.4 数据模型变更

- `Document` 增加 `task_id` 列（存量库 `init_db` 迁移补列）。
- `Document.status` 取值扩展：`pending/processing/ready/error/cancelled`。
- 进程重启兜底：lifespan 启动时把残留 `processing` 文档回置 `error`（v1 不做断点续跑）。

### 3.5 前端

- `api/documents.js`：新增 `getTask` / `cancelTask` / `streamTask`（EventSource 封装）。
- Documents 页：上传/替换秒回后渲染**任务进度卡**（文件名 + 状态徽章 + 阶段消息 + 进度条 + 取消按钮），
  `done` 事件后刷新列表并移除卡片；挂载时对 `processing` 且有 `task_id` 的文档自动重连 SSE。

## 4. 错误处理

- 任一阶段异常 → 状态机置 `failed`，error 写入 checkpoint，finalize 清理该文档已写数据。
- 向量化异常 → `blocked` 退避重试（默认 3 次、间隔 2 秒），耗尽转 `failed`。
- SSE 客户端断开不影响后台任务继续执行。
- 图内状态全纯数据（str/int/list/bytes/dict），规避 msgpack 序列化坑。

## 5. 测试计划（TDD）

1. `tests/test_task_machine.py`：合法/非法状态转移。
2. `tests/test_ingest_task_graph.py`：mock extractor/embedder 的图流转——正常全链路、
   阶段失败、取消（排队/运行中）、向量化限流 blocked→恢复、重试耗尽。
3. `tests/test_tasks_api.py`：上传秒回返回 task_id、任务查询、取消端点、SSE 流事件序列。
4. 全量回归（`USE_BGE_MODEL=false`）。

## 6. 依赖

无新增第三方依赖：`langgraph-checkpoint-sqlite`（SqliteSaver）已安装。
