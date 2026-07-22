from pathlib import Path
from typing import Dict, Optional, Type
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult


class DocumentLoaderManager:
    """DocumentLoader管理器，统一管理所有Loader"""
    
    def __init__(self):
        self._loaders: Dict[str, Type[BaseLoader]] = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self):
        """注册默认的Loader"""
        from app.services.ingestion.loaders.pdf_loader import PDFLoader
        from app.services.ingestion.loaders.text_loader import TextFileLoader
        from app.services.ingestion.loaders.markdown_loader import MarkdownLoader
        from app.services.ingestion.loaders.xmind_loader import XmindLoader
        from app.services.ingestion.loaders.word_loader import WordLoader
        from app.services.ingestion.loaders.excel_loader import ExcelLoader
        from app.services.ingestion.loaders.powerpoint_loader import PowerPointLoader
        
        self.register_loader(".pdf", PDFLoader)
        self.register_loader(".txt", TextFileLoader)
        self.register_loader(".md", MarkdownLoader)
        self.register_loader(".xmind", XmindLoader)
        self.register_loader(".docx", WordLoader)
        self.register_loader(".xlsx", ExcelLoader)
        self.register_loader(".pptx", PowerPointLoader)
    
    def register_loader(self, file_type: str, loader_class: Type[BaseLoader]):
        """注册新的Loader"""
        self._loaders[file_type] = loader_class
    
    def get_loader(self, file_type: str) -> Optional[Type[BaseLoader]]:
        """根据文件类型获取对应的Loader"""
        return self._loaders.get(file_type)
    
    def load_document(self, file_path: Path) -> DocumentResult:
        """加载文档"""
        suffix = file_path.suffix.lower()
        loader_class = self.get_loader(suffix)
        
        if not loader_class:
            raise ValueError(f"不支持的文件类型: {suffix}")
        
        loader = loader_class(str(file_path))
        return loader.load()
    
    def get_supported_types(self) -> list:
        """获取支持的文件类型列表"""
        return list(self._loaders.keys())
