import pytest
from pathlib import Path
from app.services.ingestion.loaders.excel_loader import ExcelLoader


class TestExcelLoader:
    """Excel加载器测试"""
    
    def test_load_excel_document(self, tmp_path):
        """测试加载Excel文档"""
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl not installed")
        
        file_path = tmp_path / "test.xlsx"
        wb = Workbook()
        ws = wb.active
        ws.title = "Sheet1"
        ws.append(["Name", "Age", "City"])
        ws.append(["Alice", 30, "Beijing"])
        ws.append(["Bob", 25, "Shanghai"])
        wb.save(str(file_path))
        wb.close()
        
        loader = ExcelLoader(str(file_path))
        result = loader.load()
        
        assert "Name" in result.content
        assert "Alice" in result.content
        assert result.file_type == "xlsx"
        assert "sheets" in result.structure
        assert len(result.structure["sheets"]) == 1
    
    def test_load_empty_excel(self, tmp_path):
        """测试加载空Excel文档"""
        try:
            from openpyxl import Workbook
        except ImportError:
            pytest.skip("openpyxl not installed")
        
        file_path = tmp_path / "empty.xlsx"
        wb = Workbook()
        wb.save(str(file_path))
        wb.close()
        
        loader = ExcelLoader(str(file_path))
        result = loader.load()
        
        assert result.file_type == "xlsx"
        assert "sheets" in result.structure
    
    def test_load_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises(FileNotFoundError):
            loader = ExcelLoader("/nonexistent/file.xlsx")
            loader.load()
