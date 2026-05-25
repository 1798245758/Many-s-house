import shutil
from pathlib import Path
from typing import List
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
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

def _get_deepseek_client(db: Session) -> DeepSeekClient:
    setting = db.query(Setting).filter(Setting.key == "api_key").first()
    api_key = setting.value if setting else ""
    return DeepSeekClient(api_key=api_key)

def _process_one(file: UploadFile, db: Session):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {suffix}")
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        raise Exception("文件保存失败")
    client = _get_deepseek_client(db)
    doc = ingest_document(db, save_path, file.filename or "unknown", suffix.lstrip("."), client)
    return DocumentOut.model_validate(doc).model_dump()

@router.post("/upload", response_model=ApiResponse)
def upload_documents(files: List[UploadFile] = File(...), db: Session = Depends(get_db)):
    results = []
    errors = []
    for file in files:
        try:
            data = _process_one(file, db)
            results.append(data)
        except ValueError as e:
            errors.append({"filename": file.filename, "message": str(e)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append({"filename": file.filename, "message": str(e)})
    return ApiResponse(
        code="SUCCESS" if not errors else "PARTIAL_SUCCESS",
        message=f"成功 {len(results)} 个，失败 {len(errors)} 个" if errors else "全部上传成功",
        data={"uploaded": results, "errors": errors},
    )

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
