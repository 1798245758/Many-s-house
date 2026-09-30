"""OCR 兜底测试：可用性探测 / 扫描页识别接入 / 不可用时维持报错

真实 tesseract 识别用例在引擎未安装时自动 skip（用户手动安装后生效）。
"""
from pathlib import Path
from unittest.mock import patch

import pytest

from app.services.ocr import ocr_available, ocr_image, ocr_page
from app.services.ingestion.loaders.pdf_loader import PDFLoader

PDF_DIR = Path(__file__).resolve().parents[2] / "PDF_Example"
SCANNED_PDF = PDF_DIR / "05_扫描件_纯图像无文本层_哈克贝利页29.pdf"
CHINESE_SCAN_PDF = PDF_DIR / "08_扫描件_中文合成_员工手册节选.pdf"

needs_samples = pytest.mark.skipif(not SCANNED_PDF.exists(), reason="PDF_Example 样本缺失")
needs_tesseract = pytest.mark.skipif(not ocr_available(), reason="tesseract 未安装")


class TestOcrAvailability:
    def test_unavailable_raises(self):
        """引擎不可用时 ocr_image 显式抛错（调用方可感知降级）"""
        with patch("app.services.ocr.ocr_available", return_value=False):
            with pytest.raises(RuntimeError, match="tesseract 不可用"):
                ocr_image(object())

    def test_available_locates_cmd(self):
        """可用性探测：PATH 与默认安装路径均无 tesseract 时返回 False"""
        with patch("shutil.which", return_value=None):
            assert ocr_available() is False


class TestLoaderOcrFallback:
    @needs_samples
    def test_scanned_pdf_uses_ocr_when_available(self):
        """扫描页走 OCR：产出文本、metadata 记录 ocr_pages"""
        with patch("app.services.ocr.ocr_available", return_value=True), \
             patch("app.services.ocr.ocr_page",
                   return_value="OCR识别文本") as m:
            result = PDFLoader(str(SCANNED_PDF)).load()
        assert m.called
        assert result.metadata["ocr_pages"] >= 1
        assert "OCR识别文本" in result.content

    @needs_samples
    def test_scanned_pdf_ocr_failure_still_raises(self):
        """OCR 抛异常时不吞错：整篇无文本仍显式报错"""
        with patch("app.services.ocr.ocr_available", return_value=True), \
             patch("app.services.ocr.ocr_page", side_effect=RuntimeError("boom")):
            with pytest.raises(ValueError, match="扫描件"):
                PDFLoader(str(SCANNED_PDF)).load()

    @needs_samples
    @pytest.mark.skipif(ocr_available(), reason="tesseract 已安装，扫描件会被 OCR 成功处理")
    def test_scanned_pdf_ocr_disabled_raises(self):
        """OCR_ENABLED=false 时维持原有显式报错行为"""
        with patch("app.config.OCR_ENABLED", False):
            with pytest.raises(ValueError, match="扫描件"):
                PDFLoader(str(SCANNED_PDF)).load()

    @needs_samples
    @needs_tesseract
    def test_real_tesseract_recognizes_scanned_page(self):
        """真实引擎端到端：05 样本（英文小说扫描页）应识别出可读文本"""
        result = PDFLoader(str(SCANNED_PDF)).load()
        assert result.metadata.get("ocr_pages", 0) >= 1
        text = result.content
        assert len(text) > 100
        # 该页为《哈克贝利·费恩历险记》第29章扫描页，识别应含常见英文词
        assert any(w in text.lower() for w in ("the", "and", "he", "was"))

    @pytest.mark.skipif(not CHINESE_SCAN_PDF.exists(), reason="08 合成样本缺失")
    @needs_tesseract
    def test_chinese_synthetic_scan_ocr_golden(self):
        """OCR 金标：合成中文扫描页内容确定，识别应命中关键词

        需 tesseract 已装 chi_sim 语言包；缺失时跳过。"""
        try:
            result = PDFLoader(str(CHINESE_SCAN_PDF)).load()
        except Exception as e:
            if "chi_sim" in str(e) or "language" in str(e).lower():
                pytest.skip(f"tesseract 未装 chi_sim 语言包: {e}")
            raise
        assert result.metadata.get("ocr_pages", 0) >= 1
        # 合成时写入的确定文本（OCR 容错：至少命中部分关键词）
        text = result.content
        assert any(kw in text for kw in ("员工", "手册", "公司", "国家")), \
            f"中文 OCR 未命中关键词，识别结果: {text[:80]!r}"
