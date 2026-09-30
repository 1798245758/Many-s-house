"""将 胖东来文档 目录下的企业文档切块并向量化入库

执行前会清空旧知识库数据（documents/chunks/FTS/ChromaDB向量），
保留 settings（如 api_key）与 query_history。
旧格式 .doc/.ppt 先通过 Office COM 转换为 .docx/.pptx 再入库；
若同名现代格式文件已存在则视为内容重复而跳过旧格式文件。

用法：在 backend 目录下执行  py ingest_pangdonglai.py
"""
import shutil
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from app.database import get_engine, get_session_local, Base  # noqa: E402
from app.models.document import Document  # noqa: E402
from app.models.chunk import Chunk  # noqa: E402
from app.config import UPLOAD_DIR  # noqa: E402
from app.services.ingestion.pipeline import ingest_document  # noqa: E402
from app.services import vector_store  # noqa: E402

DOC_DIR = BASE_DIR.parent / "胖东来文档"
MODERN_EXTS = {".docx", ".pptx", ".pdf", ".txt", ".md", ".xlsx"}
LEGACY_MAP = {".doc": ".docx", ".ppt": ".pptx"}


def reset_knowledge_data(db):
    """清空文档、分块与向量库（保留settings/query_history）"""
    db.query(Chunk).delete()
    db.query(Document).delete()
    db.commit()
    # 清空ChromaDB collection中的全部向量
    collection = vector_store.get_collection()
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)
    # 重建FTS索引（此时chunks表为空）
    from app.services.ingestion.indexer import build_fts_index
    build_fts_index(db.get_bind())
    print(f"旧数据已清空，Chroma剩余向量数: {vector_store.count()}")


def convert_legacy(src: Path, dst: Path):
    """Office COM 转换旧格式：.doc→.docx / .ppt→.pptx"""
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    is_doc = src.suffix.lower() == ".doc"
    app = win32com.client.Dispatch("Word.Application" if is_doc else "PowerPoint.Application")
    if is_doc:
        app.Visible = False  # PowerPoint 不允许隐藏应用窗口，仅在 Word 下设置
    try:
        if is_doc:
            doc = app.Documents.Open(str(src), ReadOnly=True)
            doc.SaveAs2(str(dst), FileFormat=16)  # wdFormatDocumentDefault
            doc.Close(False)
        else:
            pres = app.Presentations.Open(str(src), ReadOnly=True, WithWindow=False)
            pres.SaveAs(str(dst), 24)  # ppSaveAsOpenXMLPresentation
            pres.Close()
    finally:
        app.Quit()
        pythoncom.CoUninitialize()


def main():
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    db = get_session_local()()
    try:
        reset_knowledge_data(db)

        files = sorted(p for p in DOC_DIR.iterdir() if p.is_file())
        modern_stems = {p.stem for p in files if p.suffix.lower() in MODERN_EXTS}
        existing_names = {d.filename for d in db.query(Document).all()}
        print(f"发现 {len(files)} 个文件，开始解析入库...")

        ok = fail = skipped = 0
        for path in files:
            suffix = path.suffix.lower()
            try:
                if path.name in existing_names:
                    print(f"[跳过] {path.name}：已入库")
                    skipped += 1
                    continue
                if suffix in LEGACY_MAP:
                    if path.stem in modern_stems:
                        print(f"[跳过] {path.name}：已存在同名现代格式文件，视为内容重复")
                        skipped += 1
                        continue
                    new_ext = LEGACY_MAP[suffix]
                    conv_path = UPLOAD_DIR / (path.stem + new_ext)
                    print(f"[转换] {path.name} -> {conv_path.name}")
                    convert_legacy(path, conv_path)
                    source_path, file_type = conv_path, new_ext.lstrip(".")
                elif suffix in MODERN_EXTS:
                    source_path = UPLOAD_DIR / path.name
                    shutil.copy2(path, source_path)
                    file_type = suffix.lstrip(".")
                else:
                    print(f"[跳过] {path.name}：不支持的格式 {suffix}")
                    skipped += 1
                    continue

                doc = ingest_document(db, source_path, path.name, file_type)
                existing_names.add(path.name)
                ok += 1
                print(f"[完成] {path.name}：{doc.chunk_count} 个chunks")
            except Exception as e:
                fail += 1
                print(f"[失败] {path.name}：{e}")

        print("=" * 50)
        print(f"入库完成: 成功 {ok} 个, 失败 {fail} 个, 跳过 {skipped} 个")
        print(f"SQLite文档数: {db.query(Document).count()}")
        print(f"SQLite chunks数: {db.query(Chunk).count()}")
        print(f"ChromaDB向量数: {vector_store.count()}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
