"""错误诊断 Agent：读取 LangGraph CheckPoint（SqliteSaver 落盘的状态机流转历史）
+ 文档表信息 + 错误日志，采集"现场证据"后交给 LLM 分析根因。

独立于入库/问答双链路：只做只读采集与一次性 LLM 调用，不写状态。
"""
import json

from app.models.document import Document
from app.models.query_trace import QueryTrace
from app.services.tasks import runner

# checkpoint 事件中参与诊断的字段（大体积中间产物只保留统计量）
_EVENT_FIELDS = ("status", "stage", "progress", "message", "error",
                 "retry_count", "filename", "file_type", "chunk_count")
_MAX_EVENTS = 60

SYSTEM_PROMPT = (
    "你是企业知识库系统的资深运维诊断专家。用户会提供一个异步任务的现场证据："
    "状态机流转事件日志（来自 LangGraph CheckPoint）、文档元信息、以及可选的错误日志。"
    "请基于证据分析任务失败的根本原因，用 Markdown 输出三部分："
    "## 根因分析（指出最可能的原因及证据链）、"
    "## 关键证据（引用事件日志中的具体状态/错误）、"
    "## 修复建议（可执行的具体步骤）。"
    "只依据给定证据推理，证据不足时明确说明还需要什么信息，不要编造。"
)

RETRIEVAL_SYSTEM_PROMPT = (
    "你是企业知识库系统的 RAG 检索链路诊断专家。用户会提供一轮问答的多步状态机轨迹"
    "（意图/改写/检索/精排/多跳/证据校验/回答），以及每步的级别(正常/降级/错误)与错误码。"
    "请基于轨迹分析检索失败或质量降级的根本原因，用 Markdown 输出三部分："
    "## 根因分析、## 关键证据（引用具体步骤/错误码）、## 修复建议。"
    "只依据给定轨迹推理，证据不足时说明还需什么信息，不要编造。"
)


def _slim_event(values: dict) -> dict:
    ev = {k: values.get(k) for k in _EVENT_FIELDS if values.get(k) not in (None, "", 0)}
    ev.setdefault("status", values.get("status", ""))
    return ev


def _dedup(events: list) -> list:
    """去掉连续重复事件（_announce 预告与阶段完成可能产生同值快照）"""
    out = []
    for ev in events:
        if not out or ev != out[-1]:
            out.append(ev)
    return out[-_MAX_EVENTS:]


def collect_evidence(db, task_id: str, error_log: str = "") -> dict | None:
    """采集诊断现场：checkpoint 最新状态 + 流转事件日志 + 文档信息；任务不存在返回 None"""
    snapshot = runner.get_task_snapshot(task_id)
    if not snapshot:
        return None
    history = runner.get_task_history(task_id)
    doc_id = snapshot.get("doc_id")
    doc = db.query(Document).filter(Document.id == doc_id).first() if doc_id else None
    return {
        "task_id": task_id,
        "task_type": snapshot.get("task_type", ""),
        "final_status": snapshot.get("status", ""),
        "final_stage": snapshot.get("stage", ""),
        "error": snapshot.get("error", ""),
        "retry_count": snapshot.get("retry_count", 0),
        "events": _dedup([_slim_event(h) for h in history]),
        "document": {
            "id": doc.id, "filename": doc.filename, "file_type": doc.file_type,
            "file_size": doc.file_size, "status": doc.status,
        } if doc else None,
        "error_log": (error_log or "")[-4000:],  # 截断防 prompt 爆炸
    }


def _evidence_to_text(evidence: dict) -> str:
    parts = [
        f"任务ID: {evidence['task_id']}（类型: {evidence['task_type']}）",
        f"最终状态: {evidence['final_status']}，失败阶段: {evidence['final_stage'] or '未知'}",
        f"错误信息: {evidence['error'] or '无'}",
        f"重试次数: {evidence['retry_count']}",
    ]
    if evidence.get("document"):
        d = evidence["document"]
        parts.append(f"文档: {d['filename']}（{d['file_type']}, {d['file_size']}B, 状态={d['status']}）")
    parts.append("状态机流转事件日志（时间序）:")
    parts.append(json.dumps(evidence["events"], ensure_ascii=False, indent=1))
    if evidence.get("error_log"):
        parts.append(f"附加错误日志:\n{evidence['error_log']}")
    return "\n".join(parts)


def collect_query_evidence(db, trace_id: int, error_log: str = "") -> dict | None:
    """采集一轮问答的检索轨迹证据；轨迹不存在返回 None"""
    row = db.query(QueryTrace).filter(QueryTrace.id == trace_id).first()
    if not row:
        return None
    try:
        steps = json.loads(row.steps or "[]")
    except Exception:
        steps = []
    return {
        "subject_type": "query", "trace_id": row.id,
        "conversation_id": row.conversation_id or "", "question": row.question or "",
        "answer_snippet": row.answer_snippet or "", "response_type": row.response_type or "",
        "final_level": row.final_level or "normal", "final_code": row.final_code or "",
        "steps": steps, "error_log": (error_log or "")[-4000:],
    }


def _query_evidence_to_text(evidence: dict) -> str:
    parts = [
        f"会话: {evidence['conversation_id']}",
        f"问题: {evidence['question']}",
        f"响应类型: {evidence['response_type']}，最终级别: {evidence['final_level']}"
        f"（{evidence['final_code']}）",
        f"答案摘要: {evidence['answer_snippet'][:200]}",
        "多步轨迹（时间序）:",
    ]
    for s in evidence.get("steps", []):
        parts.append(f"- [{s.get('level')}/{s.get('code', '')}] {s.get('step')}: "
                     f"{s.get('message', '')} {s.get('metrics') or ''}")
    if evidence.get("error_log"):
        parts.append(f"附加错误日志:\n{evidence['error_log']}")
    return "\n".join(parts)


def analyze(evidence: dict, client) -> dict:
    """单次 LLM 调用产出诊断报告；按证据类型(query/task)分流；LLM 失败降级返回摘要"""
    is_query = evidence.get("subject_type") == "query" or "steps" in evidence
    if is_query:
        system, user = RETRIEVAL_SYSTEM_PROMPT, _query_evidence_to_text(evidence)
        base = {"trace_id": evidence.get("trace_id"),
                "final_level": evidence.get("final_level", "")}
    else:
        system, user = SYSTEM_PROMPT, _evidence_to_text(evidence)
        base = {"task_id": evidence.get("task_id"),
                "final_status": evidence.get("final_status", "")}
    try:
        analysis = client.ask(system, user)
    except Exception as e:
        return {**base, "error": evidence.get("error", ""), "analysis": "",
                "llm_error": f"LLM 诊断调用失败: {e}"}
    return {**base, "error": evidence.get("error", ""), "analysis": analysis, "llm_error": ""}
