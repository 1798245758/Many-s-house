from pathlib import Path
from typing import Any, Dict, List
from app.services.ingestion.loaders.base import BaseLoader, DocumentResult


class MarkdownLoader(BaseLoader):
    """Markdown文件加载器"""
    
    def load(self) -> DocumentResult:
        """加载Markdown文档"""
        if not self.file_path.exists():
            raise FileNotFoundError(f"文件不存在: {self.file_path}")
        
        content = self.file_path.read_text(encoding="utf-8")
        metadata = self._get_file_metadata()
        structure = self._extract_structure(content)
        
        return DocumentResult(
            content=content,
            metadata=metadata,
            structure=structure,
            source=str(self.file_path),
            file_type="md",
        )
    
    def _extract_structure(self, content: str) -> Dict[str, Any]:
        """提取Markdown结构信息"""
        sections = []
        current_section = None
        
        for line in content.split("\n"):
            if line.startswith("#"):
                level = len(line.split(" ")[0])
                title = line.lstrip("#").strip()
                current_section = {
                    "level": level,
                    "title": title,
                    "content": []
                }
                sections.append(current_section)
            elif current_section is not None:
                current_section["content"].append(line)
        
        return {
            "sections": sections,
            "heading_count": len(sections),
        }