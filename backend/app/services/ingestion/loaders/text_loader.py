from pathlib import Path
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult


class TextFileLoader(BaseLoader):
    """文本文件加载器"""
    
    def load(self) -> DocumentResult:
        """加载文本文档"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        content = self.file_path.read_text(encoding="utf-8")
        metadata = self._get_file_metadata()
        
        return DocumentResult(
            content=content,
            metadata=metadata,
            structure={},
            source=str(self.file_path),
            file_type="txt",
        )
