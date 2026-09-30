"""文档入库任务状态图：extract → clean → chunk → embed ⇄ wait_retry → index → finalize

每个阶段结束即一次 super-step，状态（含状态机取值/阶段/进度）自动写入 Checkpoint，
get_state_history 即完整的状态流转事件日志。
"""
import json
import time
from pathlib import Path

from langgraph.graph import StateGraph, START, END

from app.database import get_session_local
from app.models.chunk import Chunk
from app.models.document import Document
from app.services import vector_store
from app.services.ingestion.chunker import semantic_chunk
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.embedder import vectorize_chunks
from app.services.ingestion.extractor import extract_document
from app.services.ingestion.indexer import save_chunks
from app.services.tasks.cancel import is_cancel_requested, clear_cancel
from app.services.tasks.machine import (
    BLOCKED, CANCELLED, FAILED, RUNNING, SUCCEEDED, TERMINAL_STATES, transition,
)
from app.services.tasks.state import TaskState

# 向量化外部依赖重试配置（退避窗口期状态机置 blocked）
EMBED_MAX_RETRIES = 3
EMBED_RETRY_DELAY = 2.0

# 阶段 → 进度区间（固定权重）
STAGE_PROGRESS = {
    "extract": (0, 30),
    "clean": (30, 40),
    "chunk": (40, 50),
    "embed": (50, 90),
    "index": (90, 100),
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

# 编译后的图引用（供节点内 announce 提前写"进行中"checkpoint）
_compiled_holder = {}


def _announce(config, values: dict):
    """阶段开始时提前写一次 checkpoint，让长阶段执行期间前端能看到"XX中…"

    仅用于展示，不参与路由；失败静默降级。
    """
    graph = _compiled_holder.get("graph")
    if graph is None:
        return
    try:
        graph.update_state(config, values)
    except Exception:
        pass


def _cancelled_update(state) -> dict:
    return {"status": transition(state.get("status", RUNNING), CANCELLED),
            "message": "任务已取消"}


# ---------- 阶段工作函数（纯数据进出，DB 交互收口在辅助函数） ----------

def _save_doc_metadata(doc_id: int, metadata: dict, structure: dict):
    db = get_session_local()()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.metadata_json = json.dumps(metadata, ensure_ascii=False)
            doc.structure_json = json.dumps(structure, ensure_ascii=False)
            db.commit()
    finally:
        db.close()


def _cleanup_doc(doc_id: int, status: str):
    """失败/取消收口：清理已写 chunks 与向量，文档置对应终态"""
    db = get_session_local()()
    try:
        db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc:
            doc.status = status
            doc.chunk_count = 0
        db.commit()
    finally:
        db.close()
    try:
        vector_store.delete_by_document(doc_id)
    except Exception:
        pass


def _extract_work(state):
    file_type = state["file_type"]
    if f".{file_type}" in IMAGE_EXTENSIONS:
        raw_text = f"[图片文件: {state['filename']}]  (可上传 XMind 等思维导图导出图片，系统已保存)"
    else:
        info = extract_document(Path(state["file_path"]))
        raw_text = info["content"]
        _save_doc_metadata(state["doc_id"], info.get("metadata", {}), info.get("structure", {}))
    return {"raw_text": raw_text}


def _clean_work(state):
    return {"cleaned_text": clean_text(state["raw_text"])}


def _chunk_work(state):
    chunks = semantic_chunk(state["cleaned_text"])
    return {"chunks": chunks, "chunk_count": len(chunks)}


def _index_work(state):
    db = get_session_local()()
    try:
        save_chunks(db, state["doc_id"], state["chunks"], state["embeddings"])
    finally:
        db.close()
    return {}


def _stage_node(stage: str, running_msg: str, work, done_status: str = RUNNING):
    """通用阶段节点：取消检查 → 预告"进行中" → 执行 → 完成/失败"""
    start, end = STAGE_PROGRESS[stage]

    def node(state, config):
        if is_cancel_requested(state["task_id"]):
            return _cancelled_update(state)
        _announce(config, {"status": RUNNING, "stage": stage, "progress": start, "message": running_msg})
        try:
            updates = work(state) or {}
        except Exception as e:
            return {"status": transition(RUNNING, FAILED), "stage": stage,
                    "error": str(e), "message": f"{running_msg}失败"}
        return {"status": transition(RUNNING, done_status), "stage": stage, "progress": end,
                "message": f"{running_msg}完成", **updates}

    return node


def _embed_node(state, config):
    """向量化节点：异常时置 blocked 并递增重试计数，由条件边路由到 wait_retry"""
    if is_cancel_requested(state["task_id"]):
        return _cancelled_update(state)
    _announce(config, {"status": transition(state.get("status", RUNNING), RUNNING),
                       "stage": "embed", "progress": STAGE_PROGRESS["embed"][0], "message": "向量化中"})
    try:
        embeddings = vectorize_chunks(state["chunks"])
    except Exception as e:
        retry_count = state.get("retry_count", 0)
        if retry_count >= EMBED_MAX_RETRIES:
            return {"status": transition(state.get("status", RUNNING), FAILED), "stage": "embed",
                    "error": str(e), "message": "向量化失败"}
        return {"status": transition(state.get("status", RUNNING), BLOCKED), "stage": "embed",
                "retry_count": retry_count + 1,
                "message": f"向量化服务暂不可用，{EMBED_RETRY_DELAY:.0f}秒后重试"
                           f"（{retry_count + 1}/{EMBED_MAX_RETRIES}）"}
    return {"status": RUNNING, "stage": "embed", "progress": STAGE_PROGRESS["embed"][1],
            "message": "向量化完成", "embeddings": embeddings}


def _wait_retry_node(state, config):
    """阻塞退避：等待后回到向量化节点重试"""
    time.sleep(EMBED_RETRY_DELAY)
    return {"status": transition(state.get("status", BLOCKED), RUNNING), "message": "正在重试向量化"}


def _finalize_node(state, config):
    """所有路径收口：终态同步到文档表 + 清理 + 释放取消标志"""
    clear_cancel(state["task_id"])
    status = state.get("status")
    if status == FAILED:
        _cleanup_doc(state["doc_id"], "error")
    elif status == CANCELLED:
        _cleanup_doc(state["doc_id"], "cancelled")
    return {}


# ---------- 路由 ----------

def _after_stage(next_node: str):
    """阶段后路由：终态→收口，阻塞→退避，否则进入下一阶段（每轮显式读本轮状态）"""
    def route(state):
        status = state.get("status")
        if status in TERMINAL_STATES:
            return "finalize"
        if status == BLOCKED:
            return "wait_retry"
        return next_node
    return route


_BRANCHES = {"finalize": "finalize", "wait_retry": "wait_retry"}


def build_ingest_graph(checkpointer=None):
    builder = StateGraph(TaskState)
    builder.add_node("extract", _stage_node("extract", "解析文档", _extract_work))
    builder.add_node("clean", _stage_node("clean", "清洗文本", _clean_work))
    builder.add_node("chunk", _stage_node("chunk", "切分文本块", _chunk_work))
    builder.add_node("embed", _embed_node)
    builder.add_node("wait_retry", _wait_retry_node)
    builder.add_node("index", _stage_node("index", "写入索引", _index_work, done_status=SUCCEEDED))
    builder.add_node("finalize", _finalize_node)

    builder.add_edge(START, "extract")
    builder.add_conditional_edges("extract", _after_stage("clean"), {**_BRANCHES, "clean": "clean"})
    builder.add_conditional_edges("clean", _after_stage("chunk"), {**_BRANCHES, "chunk": "chunk"})
    builder.add_conditional_edges("chunk", _after_stage("embed"), {**_BRANCHES, "embed": "embed"})
    builder.add_conditional_edges("embed", _after_stage("index"), {**_BRANCHES, "index": "index"})
    builder.add_edge("wait_retry", "embed")
    builder.add_edge("index", "finalize")
    builder.add_edge("finalize", END)

    compiled = builder.compile(checkpointer=checkpointer)
    _compiled_holder["graph"] = compiled
    return compiled
