import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import UPLOAD_DIR
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.setting import Setting
from app.schemas.common import ApiResponse
from app.schemas.document import DocumentOut
from app.services.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/api/documents", tags=["documents"])
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}

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
    doc = ingest_document(db, save_path, file.filename or "unknown", suffix.lstrip("."))
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
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query(""),
    db: Session = Depends(get_db),
):
    query = db.query(Document)
    if search:
        query = query.filter(Document.filename.contains(search))
    total = query.count()
    docs = query.order_by(Document.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return ApiResponse(data={
        "items": [DocumentOut.model_validate(d).model_dump() for d in docs],
        "total": total,
        "page": page,
        "page_size": page_size,
    })

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


@router.put("/{doc_id}", response_model=ApiResponse)
def replace_document(doc_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return ApiResponse(code="NOT_FOUND", message="文档不存在")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return ApiResponse(code="PARAM_ERROR", message=f"不支持的文件类型: {suffix}")

    old_file = UPLOAD_DIR / doc.filename
    if old_file.exists():
        old_file.unlink()

    db.query(Chunk).filter(Chunk.document_id == doc_id).delete()
    db.flush()

    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return ApiResponse(code="SERVER_ERROR", message="文件保存失败")

    try:
        updated = ingest_document(db, save_path, file.filename or "unknown", suffix.lstrip("."), doc_id=doc_id)
    except ValueError as e:
        return ApiResponse(code="PARAM_ERROR", message=str(e))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse(code="DOC_PROCESS_ERROR", message=str(e))

    return ApiResponse(code="SUCCESS", message="文档已更新", data=DocumentOut.model_validate(updated).model_dump())
