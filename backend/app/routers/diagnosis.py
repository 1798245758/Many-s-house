"""错误诊断 API：可诊断任务列表、现场证据查看、LLM 根因分析（仅经理）"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.document import Document
from app.models.query_trace import QueryTrace
from app.models.setting import Setting
from app.schemas.common import ApiResponse
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisChatRequest, QueryAnalyzeRequest
from app.services.deepseek import DeepSeekClient
from app.services.diagnosis import agent
from app.services.diagnosis import context as diag_ctx
from app.services.role import ROLE_MANAGER, get_role

router = APIRouter(prefix="/api/diagnosis", tags=["diagnosis"])


def _manager_only(role: str):
    return None if role == ROLE_MANAGER else ApiResponse(
        code="PERMISSION_DENIED", message="仅经理可使用错误诊断")


@router.get("/tasks", response_model=ApiResponse)
def list_diagnosable_tasks(db: Session = Depends(get_db), role: str = Depends(get_role)):
    """可诊断任务：失败/已取消且留有 checkpoint thread_id 的文档"""
    denied = _manager_only(role)
    if denied:
        return denied
    docs = (db.query(Document)
            .filter(Document.status.in_(["error", "cancelled"]), Document.task_id.isnot(None))
            .order_by(Document.id.desc()).limit(50).all())
    return ApiResponse(data=[{
        "task_id": d.task_id, "doc_id": d.id, "filename": d.filename,
        "file_type": d.file_type, "doc_status": d.status, "created_at": d.created_at,
    } for d in docs])


@router.get("/tasks/{task_id}/evidence", response_model=ApiResponse)
def get_evidence(task_id: str, db: Session = Depends(get_db), role: str = Depends(get_role)):
    """查看 checkpoint 现场证据（不调 LLM）"""
    denied = _manager_only(role)
    if denied:
        return denied
    evidence = agent.collect_evidence(db, task_id)
    if not evidence:
        return ApiResponse(code="NOT_FOUND", message="任务不存在或 checkpoint 已清理")
    return ApiResponse(data=evidence)


@router.post("/analyze", response_model=ApiResponse)
def analyze_task(req: DiagnosisRequest, db: Session = Depends(get_db), role: str = Depends(get_role)):
    """错误诊断 Agent：采集 checkpoint 证据 + 错误日志，交 LLM 分析根因"""
    denied = _manager_only(role)
    if denied:
        return denied
    evidence = agent.collect_evidence(db, req.task_id, req.error_log or "")
    if not evidence:
        return ApiResponse(code="NOT_FOUND", message="任务不存在或 checkpoint 已清理")
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    if not setting or not setting.value:
        return ApiResponse(code="API_KEY_MISSING", message="请先在个人中心配置 DeepSeek API Key")
    result = agent.analyze(evidence, DeepSeekClient(api_key=setting.value))
    result["evidence"] = evidence
    if result.get("llm_error"):
        return ApiResponse(code="LLM_ERROR", message=result["llm_error"], data=result)
    return ApiResponse(data=result)


@router.get("/query-errors", response_model=ApiResponse)
def list_query_errors(level: str = "", conversation_id: str = "",
                      limit: int = Query(50, ge=1, le=200),
                      db: Session = Depends(get_db), role: str = Depends(get_role)):
    """检索/问答错误轨迹列表（按级别/会话过滤，倒序）"""
    denied = _manager_only(role)
    if denied:
        return denied
    q = db.query(QueryTrace)
    if level:
        q = q.filter(QueryTrace.final_level == level)
    if conversation_id:
        q = q.filter(QueryTrace.conversation_id == conversation_id)
    rows = q.order_by(QueryTrace.id.desc()).limit(limit).all()
    return ApiResponse(data=[{
        "trace_id": r.id, "conversation_id": r.conversation_id, "question": r.question,
        "response_type": r.response_type, "final_level": r.final_level,
        "final_code": r.final_code, "created_at": r.created_at,
    } for r in rows])


@router.get("/query-traces/{trace_id}", response_model=ApiResponse)
def get_query_trace(trace_id: int, db: Session = Depends(get_db), role: str = Depends(get_role)):
    """查看单轮检索轨迹证据（不调 LLM）"""
    denied = _manager_only(role)
    if denied:
        return denied
    ev = agent.collect_query_evidence(db, trace_id)
    if not ev:
        return ApiResponse(code="NOT_FOUND", message="轨迹不存在")
    return ApiResponse(data=ev)


@router.post("/analyze-query", response_model=ApiResponse)
def analyze_query(req: QueryAnalyzeRequest, db: Session = Depends(get_db), role: str = Depends(get_role)):
    """检索错误分析 Agent：采集轨迹证据 + 错误日志，交 LLM 分析根因"""
    denied = _manager_only(role)
    if denied:
        return denied
    ev = agent.collect_query_evidence(db, req.trace_id, req.error_log or "")
    if not ev:
        return ApiResponse(code="NOT_FOUND", message="轨迹不存在")
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    if not setting or not setting.value:
        return ApiResponse(code="API_KEY_MISSING", message="请先在个人中心配置 DeepSeek API Key")
    result = agent.analyze(ev, DeepSeekClient(api_key=setting.value))
    result["evidence"] = ev
    if result.get("llm_error"):
        return ApiResponse(code="LLM_ERROR", message=result["llm_error"], data=result)
    return ApiResponse(data=result)


@router.post("/chat", response_model=ApiResponse)
def chat_diagnosis(req: DiagnosisChatRequest, db: Session = Depends(get_db), role: str = Depends(get_role)):
    """多轮诊断追问：按 subject_type 分流（task 入库任务 / query 检索问答），共用上下文管理"""
    denied = _manager_only(role)
    if denied:
        return denied
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    if not setting or not setting.value:
        return ApiResponse(code="API_KEY_MISSING", message="请先在个人中心配置 DeepSeek API Key")

    is_query = req.subject_type == "query"
    subject_id = req.trace_id if is_query else req.task_id
    # 已有会话且主体一致时复用钉住证据（evidence=None），否则采集一次现场证据
    existing = diag_ctx._sessions.get(req.session_id) if req.session_id else None
    if (existing is not None and existing.subject_id == subject_id
            and existing.subject_type == req.subject_type):
        evidence = None
    elif is_query:
        evidence = agent.collect_query_evidence(db, req.trace_id, req.error_log or "")
        if not evidence:
            return ApiResponse(code="NOT_FOUND", message="轨迹不存在")
    else:
        evidence = agent.collect_evidence(db, req.task_id, req.error_log or "")
        if not evidence:
            return ApiResponse(code="NOT_FOUND", message="任务不存在或 checkpoint 已清理")

    client = DeepSeekClient(api_key=setting.value)
    result = diag_ctx.ask(subject_id, req.question, evidence, client, req.session_id,
                          subject_type=req.subject_type)
    if result.get("llm_error"):
        return ApiResponse(code="LLM_ERROR", message=result["llm_error"], data=result)
    return ApiResponse(data=result)
