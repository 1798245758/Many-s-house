"""OCR 兜底：Tesseract 识别扫描页图像

仅当页面既无文本层又含图像时由 PDFLoader 调用；tesseract 未安装时
ocr_available() 返回 False，调用方维持原有显式报错行为。
"""
import logging
import shutil

logger = logging.getLogger(__name__)

_DEFAULT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def ocr_available() -> bool:
    """pytesseract 已安装且 tesseract 可执行文件可定位"""
    try:
        import pytesseract
        from app.config import OCR_TESSERACT_CMD
    except ImportError:
        return False
    cmd = OCR_TESSERACT_CMD or (shutil.which("tesseract") or _DEFAULT_CMD)
    if not shutil.which(cmd):
        return False
    pytesseract.pytesseract.tesseract_cmd = cmd
    return True


def ocr_image(image, lang: str | None = None) -> str:
    """PIL 图像 → 识别文本；tesseract 不可用时抛 RuntimeError"""
    if not ocr_available():
        raise RuntimeError("tesseract 不可用，无法执行 OCR")
    import pytesseract
    from app.config import OCR_LANG
    return pytesseract.image_to_string(image, lang=lang or OCR_LANG)


def ocr_page(page, dpi: int = 200) -> str:
    """pdfplumber 页 → 渲染位图 → OCR 文本"""
    return ocr_image(page.to_image(resolution=dpi).original)
