"""P1 结构还原：双栏检测 / 页眉页脚清理 / 表格Markdown化 / 章节前缀与表格独立成块"""
import struct
from pathlib import Path

import pytest

from app.models.chunk import Chunk
from app.services.embedding import EMBEDDING_DIM
from app.services.ingestion import pipeline
from app.services.ingestion.loaders.pdf_loader import (
    PDFLoader, _column_parts, _strip_repeated_lines, _table_to_markdown)

PDF_DIR = Path(__file__).resolve().parents[2] / "PDF_Example"
TWO_COL = PDF_DIR / "02_多栏排版_LLM综述.pdf"
SINGLE_COL = PDF_DIR / "01_传统文本_规整排版_注意力论文单栏.pdf"
TABLE_PDF = PDF_DIR / "03_嵌入表格_加州裁员预警报告.pdf"

needs = pytest.mark.skipif(
    not (TWO_COL.exists() and SINGLE_COL.exists()), reason="PDF_Example 样本缺失")


class TestColumnDetection:
    @needs
    def test_two_column_page_split(self):
        """双栏页裁切为左右两栏，且各栏文本独立完整"""
        import pdfplumber
        with pdfplumber.open(str(TWO_COL)) as pdf:
            parts = _column_parts(pdf.pages[5])
        assert len(parts) == 2
        left, right = (p.extract_text() or "" for p in parts)
        left_lines = {l.strip() for l in left.splitlines() if l.strip()}
        right_lines = {l.strip() for l in right.splitlines() if l.strip()}
        assert left_lines and right_lines
        # 分栏后两栏内容不再交错：行集合互斥
        assert not (left_lines & right_lines)

    @needs
    def test_single_column_page_kept(self):
        """单栏页（含居中公式）不误判为双栏"""
        import pdfplumber
        with pdfplumber.open(str(SINGLE_COL)) as pdf:
            assert len(_column_parts(pdf.pages[1])) == 1


class TestHeaderFooterStrip:
    def test_repeated_edge_lines_removed(self):
        pages = [f"公司机密\n第{i}页正文内容讲述制度。\n{i}" for i in range(1, 7)]
        out = _strip_repeated_lines(pages)
        assert all("公司机密" not in t for t in out)
        assert all("正文内容" in t for t in out)

    def test_page_number_lines_removed(self):
        """页码每页不同但形态一致，按频率判定剔除"""
        pages = [f"正文第{i}段内容。\n第 {i} 页" for i in range(1, 9)]
        out = _strip_repeated_lines(pages)
        assert all("页" not in t.splitlines()[-1] for t in out)

    def test_short_document_untouched(self):
        pages = ["页眉\n正文"] * 3
        assert _strip_repeated_lines(pages) == pages


def test_table_to_markdown():
    md = _table_to_markdown([["名称", "人数"], ["张三", "3"], [None, "5"]])
    lines = md.splitlines()
    assert lines[0] == "| 名称 | 人数 |"
    assert lines[1] == "| --- | --- |"
    assert "| 张三 | 3 |" in md
    assert "|  | 5 |" in md


@pytest.mark.skipif(not TABLE_PDF.exists(), reason="PDF_Example 样本缺失")
def test_loader_extracts_tables_with_page():
    """真实表格 PDF：structure.tables 携带页码与 Markdown 结构"""
    result = PDFLoader(str(TABLE_PDF)).load()
    tables = result.structure["tables"]
    assert tables
    assert all({"page", "markdown"} <= set(t) for t in tables)
    assert any(t["markdown"].startswith("|") for t in tables)


def _fake_embed(texts, client=None):
    return [struct.pack(f"{EMBEDDING_DIM}f", *([0.01] * EMBEDDING_DIM)) for _ in texts]


def test_pipeline_section_prefix_and_table_chunks(db_session, monkeypatch, tmp_path):
    """章节路径前缀随内容入库；表格独立成块且 metadata 标 type=table"""
    monkeypatch.setattr(pipeline, "extract_document", lambda p: {
        "content": "",
        "metadata": {}, "source": "x.pdf", "file_type": "pdf",
        "structure": {
            "page_count": 2,
            "headings": [{"page": 1, "text": "第一章 总则"}],
            "tables": [{"page": 2, "markdown": "| 甲 | 乙 |\n| --- | --- |\n| 1 | 2 |"}],
        },
        "page_content": ["第一页规定内容。" * 60, "第二页规定内容。" * 60],
    })
    monkeypatch.setattr(pipeline, "vectorize_chunks", _fake_embed)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-fake")
    doc = pipeline.ingest_document(db_session, f, "x.pdf", "pdf")

    chunks = db_session.query(Chunk).filter_by(document_id=doc.id).all()
    table_chunks = [c for c in chunks if c.ctype == "table"]
    text_chunks = [c for c in chunks if c.ctype is None]
    # 表格块：结构完整、页码正确
    assert len(table_chunks) == 1
    assert table_chunks[0].content.startswith("| 甲 | 乙 |")
    assert table_chunks[0].page == 2
    # 正文块：两页均携带章节前缀与 section 元数据（标题跨页延续）
    assert text_chunks and all(c.content.startswith("【第一章 总则】") for c in text_chunks)
    assert all(c.section == "第一章 总则" for c in text_chunks)


def test_split_table_keeps_header_per_part():
    md = "| 甲 | 乙 |\n| --- | --- |\n" + "\n".join(f"| 行{i} | 数据{i} |" for i in range(200))
    parts = pipeline._split_table(md, max_tokens=300)
    assert len(parts) > 1
    assert all(p.splitlines()[0] == "| 甲 | 乙 |" for p in parts)
