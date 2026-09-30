import shutil
from pathlib import Path
from typing import List
from fastapi import APIRouter, UploadFile, File, Depends, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from app.database import get_db
from app.config import UPLOAD_DIR, MANAGER_KEYWORDS
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.setting import Setting
from app.schemas.common import ApiResponse
from app.schemas.document import DocumentOut
from app.services.tasks.runner import submit_task
from app.services import vector_store
from app.services.role import get_role, ROLE_MANAGER
from app.services.retrieval.page_tools import read_page, get_tables

router = APIRouter(prefix="/api/documents", tags=["documents"])
ALLOWED_EXTENSIONS = {
    ".pdf", ".txt", ".md", ".xmind",
    ".docx", ".xlsx", ".pptx",
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"
}

def detect_visibility(filename: str) -> str:
    """文件名含经理关键词的文档自动标为经理专属"""
    return "manager_only" if any(kw in filename for kw in MANAGER_KEYWORDS) else "all"

def _accept_one(file: UploadFile, db: Session):
    """同步做校验与落盘，入库处理提交为异步任务后立即返回"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件类型: {suffix}")
    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        raise Exception("文件保存失败")
    # 可见性随创建即落库，避免处理窗口期经理文档对员工可见
    doc = Document(filename=file.filename or "unknown", file_type=suffix.lstrip("."),
                   file_size=save_path.stat().st_size, status="processing",
                   visibility=detect_visibility(file.filename or ""))
    db.add(doc)
    db.commit()
    db.refresh(doc)
    task_id = submit_task("doc_ingest", {
        "doc_id": doc.id, "filename": doc.filename,
        "file_path": str(save_path), "file_type": doc.file_type,
    })
    doc.task_id = task_id
    db.commit()
    return {"filename": doc.filename, "document_id": doc.id, "task_id": task_id}

@router.post("/upload", response_model=ApiResponse)
def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    role: str = Depends(get_role),
):
    if role != ROLE_MANAGER:
        return ApiResponse(code="PERMISSION_DENIED", message="仅经理可上传文档")
    tasks = []
    errors = []
    for file in files:
        try:
            tasks.append(_accept_one(file, db))
        except ValueError as e:
            errors.append({"filename": file.filename, "message": str(e)})
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append({"filename": file.filename, "message": str(e)})
    return ApiResponse(
        code="SUCCESS" if not errors else "PARTIAL_SUCCESS",
        message=f"已提交 {len(tasks)} 个入库任务" + (f"，{len(errors)} 个文件未受理" if errors else ""),
        data={"tasks": tasks, "errors": errors},
    )

@router.get("", response_model=ApiResponse)
def list_documents(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query(""),
    db: Session = Depends(get_db),
    role: str = Depends(get_role),
):
    query = db.query(Document)
    # 员工看不到经理专属文档（total 计数同步过滤）
    if role != ROLE_MANAGER:
        query = query.filter(Document.visibility != "manager_only")
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
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
    role: str = Depends(get_role),
):
    if role != ROLE_MANAGER:
        return ApiResponse(code="PERMISSION_DENIED", message="仅经理可删除文档")
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return ApiResponse(code="NOT_FOUND", message="文档不存在")
    db.delete(doc)
    db.commit()
    # 同步删除ChromaDB中该文档的向量
    vector_store.delete_by_document(doc_id)
    file_path = UPLOAD_DIR / doc.filename
    if file_path.exists():
        file_path.unlink()
    return ApiResponse(code="SUCCESS", message="文档已删除")


class VisibilityUpdate(BaseModel):
    visibility: str  # all | manager_only


@router.put("/{doc_id}", response_model=ApiResponse)
def replace_document(
    doc_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    role: str = Depends(get_role),
):
    if role != ROLE_MANAGER:
        return ApiResponse(code="PERMISSION_DENIED", message="仅经理可替换文档")
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
    # 同步删除ChromaDB中旧chunk的向量（新向量由ingest重新写入）
    vector_store.delete_by_document(doc_id)

    save_path = UPLOAD_DIR / file.filename
    try:
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception:
        return ApiResponse(code="SERVER_ERROR", message="文件保存失败")

    # 重置文档字段并提交异步入库任务，可见性随新文件名立即生效避免窗口期泄露
    doc.filename = file.filename or "unknown"
    doc.file_type = suffix.lstrip(".")
    doc.file_size = save_path.stat().st_size
    doc.status = "processing"
    doc.chunk_count = 0
    doc.metadata_json = None
    doc.structure_json = None
    doc.visibility = detect_visibility(file.filename or "")
    try:
        task_id = submit_task("doc_ingest", {
            "doc_id": doc.id, "filename": doc.filename,
            "file_path": str(save_path), "file_type": doc.file_type,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse(code="DOC_PROCESS_ERROR", message=str(e))
    doc.task_id = task_id
    db.commit()
    return ApiResponse(code="SUCCESS", message="文档替换任务已提交",
                       data={"document_id": doc_id, "task_id": task_id})


@router.patch("/{doc_id}/visibility", response_model=ApiResponse)
def update_visibility(
    doc_id: int,
    body: VisibilityUpdate,
    db: Session = Depends(get_db),
    role: str = Depends(get_role),
):
    """仅经理可切换文档可见性"""
    if role != ROLE_MANAGER:
        return ApiResponse(code="PERMISSION_DENIED", message="仅经理可修改文档可见性")
    if body.visibility not in ("all", "manager_only"):
        return ApiResponse(code="PARAM_ERROR", message="visibility 只能为 all 或 manager_only")
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return ApiResponse(code="NOT_FOUND", message="文档不存在")
    doc.visibility = body.visibility
    db.commit()
    return ApiResponse(code="SUCCESS", message="可见性已更新", data=DocumentOut.model_validate(doc).model_dump())


def _visible_doc(db: Session, doc_id: int, role: str):
    """可见性校验：返回 (doc, None) 或 (None, 错误响应)"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        return None, ApiResponse(code="NOT_FOUND", message="文档不存在")
    if role != ROLE_MANAGER and doc.visibility == "manager_only":
        return None, ApiResponse(code="PERMISSION_DENIED", message="无权访问该文档")
    return doc, None


@router.get("/{doc_id}/pages/{page}", response_model=ApiResponse)
def read_document_page(
    doc_id: int,
    page: int,
    db: Session = Depends(get_db),
    role: str = Depends(get_role),
):
    """按页码回读原文（citation 跳转 / Agent read_page 工具）"""
    doc, err = _visible_doc(db, doc_id, role)
    if err:
        return err
    data = read_page(db, doc_id, page)
    if data is None:
        return ApiResponse(code="NOT_FOUND", message="该页无正文内容")
    return ApiResponse(data=data)


@router.get("/{doc_id}/tables", response_model=ApiResponse)
def list_document_tables(
    doc_id: int,
    page: int | None = Query(None, ge=1),
    db: Session = Depends(get_db),
    role: str = Depends(get_role),
):
    """获取文档表格块（Markdown 行列结构，可按页码过滤）"""
    doc, err = _visible_doc(db, doc_id, role)
    if err:
        return err
    return ApiResponse(data={"items": get_tables(db, doc_id, page)})
