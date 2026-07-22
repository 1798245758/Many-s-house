from pathlib import Path
from typing import Any, Dict, List
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult


class PDFLoader(BaseLoader):
    """PDF文件加载器"""
    
    def load(self) -> DocumentResult:
        """加载PDF文档"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        try:
            import pdfplumber
        except ImportError:
            raise ImportError("请安装pdfplumber: pip install pdfplumber")
        
        text_parts = []
        metadata = self._get_file_metadata()
        
        with pdfplumber.open(str(self.file_path)) as pdf:
            # 提取PDF元数据
            if pdf.metadata:
                metadata.update({
                    "title": pdf.metadata.get("Title", ""),
                    "author": pdf.metadata.get("Author", ""),
                    "subject": pdf.metadata.get("Subject", ""),
                    "creator": pdf.metadata.get("Creator", ""),
                    "producer": pdf.metadata.get("Producer", ""),
                    "page_count": len(pdf.pages),
                })
            
            # 提取每页文本
            for i, page in enumerate(pdf.pages):
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        
        content = "\n".join(text_parts)
        
        return DocumentResult(
            content=content,
            metadata=metadata,
            structure={"page_count": metadata.get("page_count", 0)},
            source=str(self.file_path),
            file_type="pdf",
        )