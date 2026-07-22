import pytest
from pathlib import Path
from app.services.ingestion.loaders.text_loader import TextFileLoader


class TestTextFileLoader:
    """文本加载器测试"""
    
    def test_load_utf8_text(self, tmp_path):
        """测试加载UTF-8编码的文本文件"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")
        
        loader = TextFileLoader(str(test_file))
        result = loader.load()
        
        assert result.content == "Hello, World!"
        assert result.file_type == "txt"
        assert result.metadata["file_name"] == "test.txt"
    
    def test_load_chinese_text(self, tmp_path):
        """测试加载中文文本文件"""
        test_file = tmp_path / "chinese.txt"
        test_file.write_text("你好，世界！", encoding="utf-8")
        
        loader = TextFileLoader(str(test_file))
        result = loader.load()
        
        assert result.content == "你好，世界！"
    
    def test_load_multiline_text(self, tmp_path):
        """测试加载多行文本文件"""
        test_file = tmp_path / "multiline.txt"
        content = "Line 1\nLine 2\nLine 3"
        test_file.write_text(content, encoding="utf-8")
        
        loader = TextFileLoader(str(test_file))
        result = loader.load()
        
        assert result.content == content
    
    def test_load_empty_file(self, tmp_path):
        """测试加载空文件"""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("", encoding="utf-8")
        
        loader = TextFileLoader(str(test_file))
        result = loader.load()
        
        assert result.content == ""
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            loader = TextFileLoader("/nonexistent/file.txt")
            loader.load()
