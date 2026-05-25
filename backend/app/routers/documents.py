import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import UPLOAD_DIR
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.setting import Setting
from app.schemas.common import ApiResponse
from app.schemas.document import DocumentOut
from app.services.ingestion.pipeline import ingest_document
from app.services.deepseek import DeepSeekClient

router = APIRouter(prefix="/api/documents", tags=["documents"])
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}

def _get_deepseek_client(db: Session) -> DeepSeekClient:
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    api_key = setting.value if setting else ""
    return DeepSeekClient(api_key=api_key)

@router.post("/upload", response_model=ApiResponse)
def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return ApiResponse(code="PARAM_ERROR", message=f"不支持的文件类型: {suffix}")
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return ApiResponse(code="SERVER_ERROR", message="文件保存失败")
    client = _get_deepseek_client(db)
    try:
        doc = ingest_document(db, save_path, file.filename or "unknown", suffix.lstrip("."), client)
    except ValueError as e:
        return ApiResponse(code="PARAM_ERROR", message=str(e))
    except Exception:
        return ApiResponse(code="DOC_PROCESS_ERROR", message="文档处理失败")
    data = DocumentOut.model_validate(doc).model_dump()
    return ApiResponse(code="SUCCESS", message="文档上传并处理成功", data=data)

@router.get("", response_model=ApiResponse)
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return ApiResponse(data=[DocumentOut.model_validate(d).model_dump() for d in docs])

@router.delete("/{doc_id}", response_model=ApiResponse)
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return ApiResponse(code="NOT_FOUND", message="文档不存在")
    db.delete(doc)
    db.commit()
    file_path = UPLOAD_DIR / doc.filename
    if file_path.exists():
        file_path.unlink()
    return ApiResponse(code="SUCCESS", message="文档已删除")
