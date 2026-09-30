from pathlib import Path
import json
from app.services.ingestion.loaders.manager import DocumentLoaderManager


_document_loader_manager = DocumentLoaderManager()


def extract_text(file_path: Path) -> str:
    """
    提取文件文本内容
    
    Args:
        file_path: 文件路径
    
    Returns:
        提取的文本内容
    
    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    result = _document_loader_manager.load_document(file_path)
    return result.content


def extract_document(file_path: Path) -> dict:
    """
    提取文档的完整信息（内容、元数据、结构）
    
    Args:
        file_path: 文件路径
    
    Returns:
        包含content、metadata、structure的字典
    
    Raises:
        ValueError: 不支持的文件类型
        FileNotFoundError: 文件不存在
    """
    result = _document_loader_manager.load_document(file_path)
    return {
        "content": result.content,
        "metadata": result.metadata,
        "structure": result.structure,
        "source": result.source,
        "file_type": result.file_type,
        "page_content": result.page_content,
    }


def get_supported_types() -> list:
    """获取支持的文件类型列表"""
    return _document_loader_manager.get_supported_types()
