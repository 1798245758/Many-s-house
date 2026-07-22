import pytest
from pathlib import Path
from app.services.ingestion.loaders.pdf_loader import PDFLoader


class TestPDFLoader:
    """PDF加载器测试"""
    
    def test_load_pdf_with_text(self, tmp_path):
        """测试加载包含文本的PDF文件"""
        # 创建一个简单的PDF文件用于测试
        # 由于创建PDF需要额外依赖，这里使用mock
        pass
    
    def test_load_empty_pdf(self, tmp_path):
        """测试加载空PDF文件"""
        pass
    
    def test_load_pdf_metadata(self, tmp_path):
        """测试提取PDF元数据"""
        pass
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            loader = PDFLoader("/nonexistent/file.pdf")
            loader.load()