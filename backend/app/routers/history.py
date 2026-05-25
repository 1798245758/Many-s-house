from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.common import ApiResponse
from app.schemas.history import HistoryOut
from app.models.query_history import QueryHistory

router = APIRouter(prefix="/api/history", tags=["history"])

@router.get("", response_model=ApiResponse)
def list_history(db: Session = Depends(get_db)):
    histories = db.query(QueryHistory).order_by(QueryHistory.created_at.desc()).all()
    return ApiResponse(data=[HistoryOut.model_validate(h).model_dump() for h in histories])

@router.delete("/{history_id}", response_model=ApiResponse)
def delete_history(history_id: int, db: Session = Depends(get_db)):
    h = db.query(QueryHistory).filter(QueryHistory.id == history_id).first()
    if not h:
        return ApiResponse(code="NOT_FOUND", message="记录不存在")
    db.delete(h)
    db.commit()
    return ApiResponse(code="SUCCESS", message="已删除")
