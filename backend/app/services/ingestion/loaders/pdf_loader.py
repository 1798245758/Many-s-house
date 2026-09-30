import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult

# 页码型页眉页脚：纯数字 / "第N页" / "Page N" / "- 3 -" 等
_PAGE_NUM_RE = re.compile(
    r"^[-–—]?\s*(第\s*\d{1,4}\s*页|\d{1,4}|Page\s+\d{1,4})\s*[-–—]?$", re.IGNORECASE)


def _column_parts(page) -> list:
    """双栏检测：词按 top 归行后，左缘起行（<0.3W）与右栏起行（0.42W~0.68W）
    均占行数三成以上才判双栏；分割点取右栏起始 x0（避开居中公式/图注造成的
    假间隙），裁切后阅读顺序=左栏整体→右栏整体"""
    W, H = page.width, page.height
    words = page.extract_words()
    if len(words) < 40:
        return [page]
    lines: Dict[int, list] = {}
    for w in words:
        lines.setdefault(round(w["top"]), []).append(w)
    starts = [min(w["x0"] for w in ws) for ws in lines.values()]
    if len(starts) < 8:
        return [page]
    # 密集行（≥3词）占比过低 → 图表/散点页（如注意力热力图的旋转标签），
    # 非正文双栏；单靠起始 x 分布会误判，需正文行密度达标才继续
    if sum(1 for ws in lines.values() if len(ws) >= 3) < 0.4 * len(lines):
        return [page]
    left = sum(1 for x in starts if x < 0.3 * W)
    right_starts = [x for x in starts if 0.42 * W <= x < 0.68 * W]
    need = max(5, 0.3 * len(starts))
    if left < need or len(right_starts) < need:
        return [page]
    split = min(right_starts) - 4
    # 跨越分割线的词应极少（仅通栏标题/公式等），否则为误判
    if sum(1 for w in words if w["x0"] < split < w["x1"]) > 0.05 * len(words):
        return [page]
    return [page.crop((0, 0, split, H)), page.crop((split, 0, W, H))]


def _page_headings(part, body_size: float) -> List[str]:
    """标题识别（字号启发式）：字号显著大于正文、行短且不以句读结尾的行"""
    if body_size <= 0:
        return []
    words = part.extract_words(extra_attrs=["size"])
    lines: Dict[int, list] = {}
    for w in words:
        lines.setdefault(round(w["top"]), []).append(w)
    out = []
    for top in sorted(lines):
        ws = sorted(lines[top], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws).strip()
        size = max(w.get("size") or 0 for w in ws)
        if (size >= body_size * 1.15 and len(text) <= 60
                and not text.endswith(("。", ".", "，", ",", "；", ";"))):
            out.append(text)
    return out


def _table_to_markdown(rows) -> str:
    """表格 → Markdown（首行视作表头），单元格内换行压平"""
    rows = [[" ".join((c or "").split()) for c in row] for row in rows]
    if not rows or not rows[0]:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    md = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    md += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(md)


def _strip_repeated_lines(pages: List[str]) -> List[str]:
    """页眉页脚清理：多数页面顶/底重复出现的行整篇剔除；
    页码型行（每页不同但形态一致）按出现频率单独判定"""
    if len(pages) < 4:
        return pages
    counter: Counter = Counter()
    edges: List[set] = []
    num_pages = 0
    for t in pages:
        lines = [l.strip() for l in t.splitlines() if l.strip()]
        edge = set(lines[:2] + lines[-2:]) if lines else set()
        edges.append(edge)
        counter.update(edge)
        if any(_PAGE_NUM_RE.match(l) for l in edge):
            num_pages += 1
    threshold = max(3, len(pages) // 2)
    bad = {l for l, c in counter.items() if c >= threshold}
    strip_nums = num_pages >= max(3, int(len(pages) * 0.6))
    if not bad and not strip_nums:
        return pages
    out = []
    for t, edge in zip(pages, edges):
        keep = [l for l in t.splitlines()
                if l.strip() not in bad
                and not (strip_nums and l.strip() in edge
                         and _PAGE_NUM_RE.match(l.strip()))]
        out.append("\n".join(keep))
    return out


class PDFLoader(BaseLoader):
    """PDF文件加载器：逐页文本（双栏感知）+ 表格结构化 + 标题骨架 + 页眉页脚清理"""

    def load(self) -> DocumentResult:
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        try:
            import pdfplumber
        except ImportError:
            raise ImportError("请安装pdfplumber: pip install pdfplumber")

        metadata = self._get_file_metadata()
        page_content: List[str] = []
        headings: List[dict] = []
        tables: List[dict] = []
        image_pages = 0
        ocr_pages = 0

        with pdfplumber.open(str(self.file_path)) as pdf:
            if pdf.metadata:
                metadata.update({
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "producer": pdf.metadata.get("Producer", ""),
                })

            # 正文字号基准：前几页字符字号中位数（标题=显著大于该基准）
            sizes = [c["size"] for p in pdf.pages[:5] for c in p.chars[:2000]]
            body_size = statistics.median(sizes) if sizes else 0.0

            # 逐页提取：双栏页按左栏→右栏顺序拼接，保留页码对应关系
            from app.config import OCR_ENABLED
            from app.services.ocr import ocr_available, ocr_page
            use_ocr = OCR_ENABLED and ocr_available()
            for i, page in enumerate(pdf.pages, start=1):
                parts = _column_parts(page)
                texts = [(pt.extract_text() or "").strip() for pt in parts]
                if not any(texts) and page.images:
                    image_pages += 1
                    if use_ocr:  # 扫描件兜底：无文本层且有图像 → OCR 识别
                        try:
                            texts = [ocr_page(page).strip()]
                            ocr_pages += 1
                        except Exception as e:
                            import logging
                            logging.getLogger(__name__).warning(
                                f"第{i}页 OCR 失败: {e}")
                page_content.append("\n".join(t for t in texts if t))
                for pt in parts:
                    for h in _page_headings(pt, body_size):
                        headings.append({"page": i, "text": h})
                for tb in page.extract_tables():
                    md = _table_to_markdown(tb)
                    if md:
                        tables.append({"page": i, "markdown": md})

        page_content = _strip_repeated_lines(page_content)

        # 无文本层不再静默产出 0 chunks：OCR 兜底仍无文本时显式报错
        if page_content and not any(t.strip() for t in page_content):
            if image_pages:
                raise ValueError(
                    f"检测到扫描件PDF（无文本层，{image_pages}页为纯图像），"
                    f"OCR兜底未产出文本（tesseract 未安装或识别失败）: {self.file_path.name}")
            raise ValueError(f"PDF无可提取文本: {self.file_path.name}")

        metadata["page_count"] = len(page_content)
        if ocr_pages:
            metadata["ocr_pages"] = ocr_pages
        return DocumentResult(
            content="\n".join(t for t in page_content if t.strip()),
            metadata=metadata,
            structure={"page_count": len(page_content),
                       "headings": headings, "tables": tables},
            source=str(self.file_path),
            file_type="pdf",
            page_content=page_content,
        )
