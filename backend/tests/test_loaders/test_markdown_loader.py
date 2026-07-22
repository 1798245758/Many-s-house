import pytest
from pathlib import Path
from app.services.ingestion.loaders.markdown_loader import MarkdownLoader


class TestMarkdownLoader:
    """Markdown加载器测试"""
    
    def test_load_simple_markdown(self, tmp_path):
        """测试加载简单Markdown文件"""
        test_file = tmp_path / "test.md"
        content = "# Title\n\nThis is content."
        test_file.write_text(content, encoding="utf-8")
        
        loader = MarkdownLoader(str(test_file))
        result = loader.load()
        
        assert result.content == content
        assert result.file_type == "md"
    
    def test_load_markdown_with_structure(self, tmp_path):
        """测试加载包含结构的Markdown文件"""
        test_file = tmp_path / "structured.md"
        content = """# Main Title

## Section 1

Content of section 1.

## Section 2

Content of section 2.

### Subsection 2.1

Subsection content.
"""
        test_file.write_text(content, encoding="utf-8")
        
        loader = MarkdownLoader(str(test_file))
        result = loader.load()
        
        assert result.content == content
        assert "sections" in result.structure
    
    def test_load_empty_markdown(self, tmp_path):
        """测试加载空Markdown文件"""
        test_file = tmp_path / "empty.md"
        test_file.write_text("", encoding="utf-8")
        
        loader = MarkdownLoader(str(test_file))
        result = loader.load()
        
        assert result.content == ""
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            loader = MarkdownLoader("/nonexistent/file.md")
            loader.load()