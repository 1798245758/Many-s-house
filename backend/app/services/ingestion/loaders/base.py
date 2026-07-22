from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DocumentResult:
    """统一的文档处理结果格式"""
    content: str                          # 文本内容
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    structure: Dict[str, Any] = field(default_factory=dict)  # 结构信息
    source: str = ""                      # 文件来源
    file_type: str = ""                   # 文件类型
    page_content: List[str] = field(default_factory=list)    # 分页内容（用于PDF等）


class BaseLoader(ABC):
    """基础加载器抽象类"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
    
    @abstractmethod
    def load(self) -> DocumentResult:
        """加载文档并返回DocumentResult"""
        pass
    
    def _get_file_metadata(self) -> Dict[str, Any]:
        """获取文件基础元数据"""
        stat = self.file_path.stat()
        return {
            "source": str(self.file_path),
            "file_name": self.file_path.name,
            "file_size": stat.st_size,
            "file_type": self.file_path.suffix.lower(),
        }
