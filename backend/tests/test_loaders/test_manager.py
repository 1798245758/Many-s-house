import pytest
from pathlib import Path
from app.services.ingestion.loaders.manager import DocumentLoaderManager


class TestDocumentLoaderManager:
    """DocumentLoaderManager测试"""
    
    def test_get_loader_for_pdf(self):
        """测试获取PDF加载器"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".pdf")
        assert loader is not None
    
    def test_get_loader_for_txt(self):
        """测试获取文本加载器"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".txt")
        assert loader is not None
    
    def test_get_loader_for_md(self):
        """测试获取Markdown加载器"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".md")
        assert loader is not None
    
    def test_get_loader_for_xmind(self):
        """测试获取XMind加载器"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".xmind")
        assert loader is not None
    
    def test_get_loader_for_docx(self):
        """测试获取Word加载器"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".docx")
        assert loader is not None
    
    def test_get_loader_for_xlsx(self):
        """测试获取Excel加载器"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".xlsx")
        assert loader is not None
    
    def test_get_loader_for_pptx(self):
        """测试获取PowerPoint加载器"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".pptx")
        assert loader is not None
    
    def test_get_loader_for_unsupported_type(self):
        """测试获取不支持的文件类型"""
        manager = DocumentLoaderManager()
        loader = manager.get_loader(".xyz")
        assert loader is None
    
    def test_load_document_with_txt(self, tmp_path):
        """测试使用Manager加载文本文件"""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!", encoding="utf-8")
        
        manager = DocumentLoaderManager()
        result = manager.load_document(test_file)
        
        assert result.content == "Hello, World!"
        assert result.file_type == "txt"
    
    def test_load_document_with_unsupported_type(self, tmp_path):
        """测试加载不支持的文件类型"""
        test_file = tmp_path / "test.xyz"
        test_file.write_text("content", encoding="utf-8")
        
        manager = DocumentLoaderManager()
        with pytest.raises(ValueError) as excinfo:
            manager.load_document(test_file)
        assert "不支持的文件类型" in str(excinfo.value)
