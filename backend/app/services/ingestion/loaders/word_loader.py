from pathlib import Path
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult


class WordLoader(BaseLoader):
    """Word文件加载器"""

    def load(self) -> DocumentResult:
        """加载Word文档"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")

        try:
            import docx2txt
        except ImportError:
            raise ImportError("请安装docx2txt: pip install docx2txt")

        content = docx2txt.process(str(self.file_path))
        metadata = self._get_file_metadata()

        try:
            from docx import Document
            doc = Document(str(self.file_path))
            core_props = doc.core_properties
            metadata.update({
                "title": core_props.title or "",
                "author": core_props.author or "",
                "subject": core_props.subject or "",
                "created": str(core_props.created) if core_props.created else "",
                "modified": str(core_props.modified) if core_props.modified else "",
            })
        except Exception:
            pass

        return DocumentResult(
            content=content,
            metadata=metadata,
            structure={},
            source=str(self.file_path),
            file_type="docx",
        )
