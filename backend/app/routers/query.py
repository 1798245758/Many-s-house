from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
import json
from app.database import get_db
from app.schemas.common import ApiResponse
from app.schemas.query import QueryRequest
from app.models.query_history import QueryHistory
from app.models.query_trace import QueryTrace
from app.models.setting import Setting
from app.services.deepseek import DeepSeekClient
from app.services.retrieval.generator import generate_answer
from app.services.retrieval.agent_chain import generate_answer_via_chain
from app.services.role import get_role, ROLE_MANAGER, manager_doc_ids

router = APIRouter(prefix="/api", tags=["query"])


def _record_e5(db: Session, req: QueryRequest, chain_err, fallback_err=None):
    """整轮图执行失败落一条 E5 轨迹（含失败详情）；写库异常静默跳过，绝不阻断响应。"""
    try:
        detail = f"图执行失败: {chain_err}"
        detail += (f"；旧流水线兜底也失败: {fallback_err}" if fallback_err is not None
                   else "；已回退旧流水线兜底")
        db.add(QueryTrace(
            conversation_id=req.conversation_id or "default",
            question=req.question, answer_snippet="",
            response_type="error", final_level="error", final_code="E5",
            steps=json.dumps([{"step": "graph", "level": "error", "code": "E5",
                               "message": detail}], ensure_ascii=False)))
        db.commit()
    except Exception:
        db.rollback()


@router.post("/query", response_model=ApiResponse)
def ask_query(req: QueryRequest, db: Session = Depends(get_db), role: str = Depends(get_role)):
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    if not setting or not setting.value:
        return ApiResponse(code="API_KEY_MISSING", message="请先在个人中心配置 DeepSeek API Key")
    try:
        client = DeepSeekClient(api_key=setting.value)
    except ValueError:
        return ApiResponse(code="API_KEY_MISSING", message="API Key 无效")
    try:
        # 新架构：意图关卡 + LCEL 链式 RAG Chain（员工角色检索排除经理专属文档）；
        # conversation_id 携带会话短期记忆（未传时后端兜底同一默认会话）
        result = generate_answer_via_chain(db, req.question, client, role=role,
                                           conversation_id=req.conversation_id)
    except Exception as chain_err:
        # Chain 链路失败时，回退到旧的固定流水线（同样按角色排除经理专属文档）
        try:
            exclude_ids = None if role == ROLE_MANAGER else manager_doc_ids(db)
            result = generate_answer(db, req.question, client, exclude_doc_ids=exclude_ids)
            # 图执行失败但旧流水线兜底成功：仍落一条 E5（链路级故障需可诊断），不阻断响应
            _record_e5(db, req, chain_err)
        except Exception as e:
            # 旧流水线回退也失败：落 E5（携双失败详情）后返回 LLM_ERROR
            _record_e5(db, req, chain_err, fallback_err=e)
            return ApiResponse(code="LLM_ERROR", message=f"LLM 调用失败: {str(e)} (Chain: {chain_err})")
    history = QueryHistory(
        query_text=req.question,
        answer_text=result.answer,
        sources=str([s.model_dump() for s in result.sources]),
    )
    db.add(history)
    db.commit()
    return ApiResponse(code="SUCCESS", message="查询成功", data=result.model_dump())
