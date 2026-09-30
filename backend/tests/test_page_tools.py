"""P2-9 按需回读工具：read_page / get_tables"""
import struct

import pytest

from app.services.embedding import EMBEDDING_DIM
from app.services.ingestion import pipeline
from app.services.retrieval.page_tools import read_page, get_tables


def _fake_embed(texts, client=None):
    return [struct.pack(f"{EMBEDDING_DIM}f", *([0.01] * EMBEDDING_DIM)) for _ in texts]


@pytest.fixture
def ingested_doc(db_session, monkeypatch, tmp_path):
    """两页正文 + 第2页一张表格的伪造 PDF 入库结果"""
    monkeypatch.setattr(pipeline, "extract_document", lambda p: {
        "content": "", "metadata": {}, "source": "x.pdf", "file_type": "pdf",
        "structure": {"page_count": 2, "headings": [],
                      "tables": [{"page": 2, "markdown": "| 甲 | 乙 |\n| --- | --- |\n| 1 | 2 |"}]},
        "page_content": ["第一页正文内容。" * 60, "第二页正文内容。" * 60],
    })
    monkeypatch.setattr(pipeline, "vectorize_chunks", _fake_embed)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-fake")
    return pipeline.ingest_document(db_session, f, "x.pdf", "pdf")


def test_read_page_returns_only_that_page(db_session, ingested_doc):
    data = read_page(db_session, ingested_doc.id, 2)
    assert data["page"] == 2
    assert "第二页正文内容" in data["content"]
    assert "第一页" not in data["content"]
    assert data["chunk_ids"]


def test_read_page_missing_returns_none(db_session, ingested_doc):
    assert read_page(db_session, ingested_doc.id, 99) is None


def test_get_tables_filters_by_page(db_session, ingested_doc):
    items = get_tables(db_session, ingested_doc.id)
    assert len(items) == 1
    assert items[0]["page"] == 2
    assert items[0]["markdown"].startswith("| 甲 |")
    assert get_tables(db_session, ingested_doc.id, page=2) == items
    assert get_tables(db_session, ingested_doc.id, page=1) == []
