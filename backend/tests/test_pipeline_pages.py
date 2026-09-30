"""P0 页级溯源：逐页切块 → chunk.metadata_json 携带页码 → Chunk.page 可读"""
import struct

import pytest

from app.models.chunk import Chunk
from app.services.embedding import EMBEDDING_DIM
from app.services.ingestion import pipeline


def _fake_embed(texts, client=None):
    return [struct.pack(f"{EMBEDDING_DIM}f", *([0.01] * EMBEDDING_DIM)) for _ in texts]


@pytest.fixture
def fake_pdf_env(monkeypatch, tmp_path):
    """伪造两页 PDF 的提取结果与向量化，隔离真实文件与 BGE 模型"""
    page1, page2 = "第一页的规定内容。" * 60, "第二页的规定内容。" * 60
    monkeypatch.setattr(pipeline, "extract_document", lambda p: {
        "content": page1 + "\n" + page2,
        "metadata": {}, "structure": {"page_count": 2},
        "source": "x.pdf", "file_type": "pdf",
        "page_content": [page1, page2],
    })
    monkeypatch.setattr(pipeline, "vectorize_chunks", _fake_embed)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-fake")
    return f


def test_chunks_carry_page_numbers(db_session, fake_pdf_env):
    doc = pipeline.ingest_document(db_session, fake_pdf_env, "x.pdf", "pdf")
    chunks = (db_session.query(Chunk).filter_by(document_id=doc.id)
              .order_by(Chunk.chunk_index).all())
    assert chunks
    assert {c.page for c in chunks} == {1, 2}
    # 页码与内容严格对应：第1页的块只含第一页文本
    for c in chunks:
        body = "第一页" if c.page == 1 else "第二页"
        other = "第二页" if c.page == 1 else "第一页"
        assert body in c.content and other not in c.content


def test_non_pdf_documents_page_is_none(db_session, monkeypatch, tmp_path):
    """无分页信息的文档（txt 等）页码为 None，不破坏原有链路"""
    monkeypatch.setattr(pipeline, "extract_document", lambda p: {
        "content": "纯文本文档内容。" * 100,
        "metadata": {}, "structure": {},
        "source": "a.txt", "file_type": "txt", "page_content": [],
    })
    monkeypatch.setattr(pipeline, "vectorize_chunks", _fake_embed)
    f = tmp_path / "a.txt"
    f.write_text("x", encoding="utf-8")
    doc = pipeline.ingest_document(db_session, f, "a.txt", "txt")
    chunks = db_session.query(Chunk).filter_by(document_id=doc.id).all()
    assert chunks and all(c.page is None for c in chunks)


def test_scanned_pdf_marks_document_error(db_session, monkeypatch, tmp_path):
    """扫描件报错 → 文档状态 error，而非静默 ready"""
    def _raise(p):
        raise ValueError("检测到扫描件PDF（无文本层，1页为纯图像），需OCR处理: s.pdf")
    monkeypatch.setattr(pipeline, "extract_document", _raise)
    f = tmp_path / "s.pdf"
    f.write_bytes(b"%PDF-fake")
    with pytest.raises(ValueError, match="扫描件"):
        pipeline.ingest_document(db_session, f, "s.pdf", "pdf")
    db_session.refresh(doc := db_session.query(pipeline.Document).first())
    assert doc.status == "error"
    assert doc.chunk_count == 0
