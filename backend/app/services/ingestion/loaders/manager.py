from typing import Dict, Type
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult


class DocumentLoaderManager:
    """文档加载器管理器 - 注册和管理各种Loader"""
    
    _loaders: Dict[str, Type[BaseLoader]] = {}
    
    @classmethod
    def register(cls, file_type: str, loader_class: Type[BaseLoader]) -> None:
        """注册文件类型对应的Loader"""
        cls._loaders[file_type] = loader_class
    
    @classmethod
    def get_loader(cls, file_type: str) -> Type[BaseLoader]:
        """获取文件类型对应的Loader"""
        if file_type not in cls._loaders:
            raise ValueError(f"No loader registered for file type: {file_type}")
        return cls._loaders[file_type]
    
    @classmethod
    def load(cls, file_path: str, file_type: str) -> 'DocumentResult':
        """使用对应的Loader加载文档"""
        loader_class = cls.get_loader(file_type)
        loader = loader_class(file_path)
        return loader.load()
