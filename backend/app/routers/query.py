from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.common import ApiResponse
from app.schemas.query import QueryRequest
from app.models.query_history import QueryHistory
from app.models.setting import Setting
from app.services.deepseek import DeepSeekClient
from app.services.retrieval.generator import generate_answer
from app.services.retrieval.agent_chain import generate_answer_via_chain
from app.services.role import get_role, ROLE_MANAGER, manager_doc_ids

router = APIRouter(prefix="/api", tags=["query"])

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
        # 新架构：意图关卡 + LCEL 链式 RAG Chain（员工角色检索排除经理专属文档）
        result = generate_answer_via_chain(db, req.question, client, role=role)
    except Exception as chain_err:
        # Chain 链路失败时，回退到旧的固定流水线（同样按角色排除经理专属文档）
        try:
            exclude_ids = None if role == ROLE_MANAGER else manager_doc_ids(db)
            result = generate_answer(db, req.question, client, exclude_doc_ids=exclude_ids)
        except Exception as e:
            return ApiResponse(code="LLM_ERROR", message=f"LLM 调用失败: {str(e)} (Chain: {chain_err})")
    history = QueryHistory(
        query_text=req.question,
        answer_text=result.answer,
        sources=str([s.model_dump() for s in result.sources]),
    )
    db.add(history)
    db.commit()
    return ApiResponse(code="SUCCESS", message="查询成功", data=result.model_dump())
