"""P2-10 评测黄金集：以 PDF_Example 真实样本量化解析链路指标

指标1 页码溯源准确率：每个 chunk 的内容必须完整落在其标注页的原文内（目标 100%）
指标2 表格行列保真度：同一表格各行列数一致（Markdown 结构不被拍平）
指标3 双栏阅读顺序：分栏后交错次数归零（朴素提取时左右栏行严重交错）
指标4 页眉页脚清理：跨页重复页眉（NIST 报告）清理后零残留
指标5 单栏不误切：单栏规整排版（含图表散点页）不被误判为双栏
扫描件识别由 test_pdf_loader.py::test_scanned_pdf_raises 覆盖（05 样本）。
"""
import re
from pathlib import Path

import pytest

from app.services.ingestion.chunker import semantic_chunk
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.loaders.pdf_loader import PDFLoader, _column_parts

PDF_DIR = Path(__file__).resolve().parents[2] / "PDF_Example"
TABLE_PDF = PDF_DIR / "03_嵌入表格_加州裁员预警报告.pdf"
CONTRACT_PDF = PDF_DIR / "06_签名栏合同_govuk犯罪合同签署样本.pdf"
TWO_COL_PDF = PDF_DIR / "02_多栏排版_LLM综述.pdf"
SINGLE_COL_PDF = PDF_DIR / "01_传统文本_规整排版_注意力论文单栏.pdf"
HEADER_FOOTER_PDF = PDF_DIR / "07_页眉页脚_NIST_SP800-34r1.pdf"

needs_samples = pytest.mark.skipif(
    not (TABLE_PDF.exists() and CONTRACT_PDF.exists() and TWO_COL_PDF.exists()),
    reason="PDF_Example 样本缺失")


def _needs(path: Path):
    return pytest.mark.skipif(not path.exists(), reason=f"{path.name} 样本缺失")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


@needs_samples
@pytest.mark.parametrize("pdf_path", [TABLE_PDF, CONTRACT_PDF], ids=["表格报告", "签名合同"])
def test_page_traceability_accuracy(pdf_path):
    """指标1：逐页切块的每个 chunk 内容都完整包含在其标注页原文中"""
    result = PDFLoader(str(pdf_path)).load()
    pages = {i: _norm(t) for i, t in enumerate(result.page_content, 1)}
    chunks = [(pno, c) for pno, t in enumerate(result.page_content, 1)
              for c in semantic_chunk(clean_text(t))]
    assert chunks, "样本应产出至少一个 chunk"
    bad = [c[:40] for pno, c in chunks if _norm(c) not in pages[pno]]
    assert not bad, f"{pdf_path.name}: {len(bad)}/{len(chunks)} 个块页码与内容不符: {bad[:3]}"


@needs_samples
def test_table_structure_fidelity():
    """指标2：提取的每张表格行列结构一致（Markdown 不被拍平），表头非空

    注：不断言表头单元格必须出自 extract_text 页文本——表格与正文是两条
    提取路径，03 样本的字符间距异常导致部分标题仅 extract_tables 能拿到，
    这恰是表格单独结构化的价值所在。"""
    result = PDFLoader(str(TABLE_PDF)).load()
    tables = result.structure["tables"]
    assert tables, "表格样本应至少提取一张表"
    for t in tables:
        rows = [r for r in t["markdown"].splitlines() if r.startswith("|")]
        assert len(rows) >= 3, f"第{t['page']}页表格应含表头+分隔+至少一行数据"
        assert len({r.count("|") for r in rows}) == 1, f"第{t['page']}页表格列数不一致"
        header = [c.strip() for c in rows[0].strip("|").split("|")]
        assert any(header), f"第{t['page']}页表格表头全空"


@needs_samples
def test_two_column_reading_order():
    """指标3：朴素提取左右栏行交错严重；分栏重组后交错次数为 0"""
    import pdfplumber

    def alternations(seq, left_set, right_set):
        tags = ["L" if l in left_set else "R" for l in seq
                if l in left_set or l in right_set]
        return sum(1 for a, b in zip(tags, tags[1:]) if a != b)

    with pdfplumber.open(str(TWO_COL_PDF)) as pdf:
        page = pdf.pages[5]
        parts = _column_parts(page)
        assert len(parts) == 2, "双栏样本页应被检出"
        left, right = (p.extract_text() or "" for p in parts)

    left_set = {l.strip() for l in left.splitlines() if l.strip()}
    right_set = {l.strip() for l in right.splitlines() if l.strip()}
    naive = [l.strip() for l in (page.extract_text() or "").splitlines() if l.strip()]
    # 朴素按行提取：左右栏内容交错出现
    assert alternations(naive, left_set, right_set) > 5
    # 分栏重组后：左栏整体在前、右栏整体在后，仅剩交界处 1 次合法切换
    assert alternations([l.strip() for l in (left + "\n" + right).splitlines() if l.strip()],
                        left_set, right_set) <= 1


@needs_samples
def test_double_column_detected_multi_page():
    """指标3补充：真双栏正文页应被稳定检出（不止单页）"""
    import pdfplumber
    with pdfplumber.open(str(TWO_COL_PDF)) as pdf:
        detected = [i for i, p in enumerate(pdf.pages[:8], 1)
                    if len(_column_parts(p)) == 2]
    assert len(detected) >= 2, f"双栏样本检出页过少: {detected}"


@_needs(SINGLE_COL_PDF)
def test_single_column_not_split():
    """指标5：单栏规整排版（含图表散点页）不应被误判为双栏

    回归护栏：注意力论文第15页为热力图可视化（旋转单词标签），
    仅凭起始 x 分布会被误切；密集行占比守卫应拦住该误判。"""
    import pdfplumber
    with pdfplumber.open(str(SINGLE_COL_PDF)) as pdf:
        split_pages = [i for i, p in enumerate(pdf.pages, 1)
                       if len(_column_parts(p)) == 2]
    assert not split_pages, f"单栏样本被误切为双栏的页: {split_pages}"


@_needs(HEADER_FOOTER_PDF)
def test_repeated_header_stripped():
    """指标4：跨页重复页眉（NIST 报告每页顶部标题）清理后零残留"""
    header = "CONTINGENCY PLANNING GUIDE FOR FEDERAL INFORMATION SYSTEMS"
    import pdfplumber
    with pdfplumber.open(str(HEADER_FOOTER_PDF)) as pdf:
        raw_hits = sum(1 for p in pdf.pages[10:40]
                       if header in (p.extract_text() or ""))
    assert raw_hits >= 15, f"样本重复页眉过少({raw_hits})，无法验证清理"
    result = PDFLoader(str(HEADER_FOOTER_PDF)).load()
    residual = [i for i, t in enumerate(result.page_content, 1) if header in t]
    assert not residual, f"重复页眉未清理，残留页: {residual[:5]}"
