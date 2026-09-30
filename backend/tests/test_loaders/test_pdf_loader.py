import pytest
from pathlib import Path
from app.services.ingestion.loaders.pdf_loader import PDFLoader
from app.services.ocr import ocr_available

# 仓库根目录下的真实 PDF 黄金样本（多栏/表格/扫描件/签名合同等形态）
PDF_DIR = Path(__file__).resolve().parents[3] / "PDF_Example"
TEXT_PDF = PDF_DIR / "06_签名栏合同_govuk犯罪合同签署样本.pdf"
SCANNED_PDF = PDF_DIR / "05_扫描件_纯图像无文本层_哈克贝利页29.pdf"

needs_samples = pytest.mark.skipif(
    not (TEXT_PDF.exists() and SCANNED_PDF.exists()), reason="PDF_Example 样本缺失")


class TestPDFLoader:
    """PDF加载器测试"""

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            loader = PDFLoader("/nonexistent/file.pdf")
            loader.load()

    @needs_samples
    def test_page_content_populated(self):
        """逐页文本保留页码对应关系（页级溯源地基）"""
        result = PDFLoader(str(TEXT_PDF)).load()
        assert len(result.page_content) == 4
        assert result.structure["page_count"] == 4
        assert any(t.strip() for t in result.page_content)
        # content 与 page_content 一致（非空页拼接）
        assert result.content == "\n".join(t for t in result.page_content if t.strip())

    @needs_samples
    @pytest.mark.skipif(ocr_available(), reason="tesseract 已安装，扫描件会被 OCR 成功处理（见 test_ocr.py）")
    def test_scanned_pdf_raises(self):
        """扫描件（无文本层纯图像）且 OCR 不可用时显式报错，不再静默产出 0 chunks"""
        with pytest.raises(ValueError, match="扫描件"):
            PDFLoader(str(SCANNED_PDF)).load()

    def test_empty_pdf_raises(self, tmp_path):
        """无任何文本且无图像的 PDF 也显式报错"""
        from pypdf import PdfWriter
        f = tmp_path / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(f, "wb") as fh:
            writer.write(fh)
        with pytest.raises(ValueError, match="无可提取文本"):
            PDFLoader(str(f)).load()
