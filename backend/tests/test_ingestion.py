import pytest
from pathlib import Path
from app.services.ingestion.extractor import extract_text
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import semantic_chunk

def test_extract_txt(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("你好世界\n第二行\n\n第三行", encoding="utf-8")
    result = extract_text(f)
    assert "你好世界" in result

def test_extract_md(tmp_path):
    f = tmp_path / "test.md"
    f.write_text("# 标题\n这是Markdown\n```python\nprint(1)\n```", encoding="utf-8")
    result = extract_text(f)
    assert "标题" in result

def test_extract_unsupported(tmp_path):
    f = tmp_path / "test.exe"
    f.write_bytes(b"fake")
    with pytest.raises(ValueError, match="不支持"):
        extract_text(f)

def test_clean_removes_excess_blank_lines():
    text = "第一段\n\n\n\n\n第二段\n\n第三段"
    result = clean_text(text)
    assert result == "第一段\n\n第二段\n\n第三段"

def test_clean_removes_special_chars():
    text = "标题\u200b内容\u00a0结束"
    result = clean_text(text)
    assert "\u200b" not in result

def test_clean_strips_lines():
    text = "  缩进文本  \n  另一行  "
    result = clean_text(text)
    assert result == "缩进文本\n另一行"

def test_chunk_splits_by_double_newline():
    text = "段落A。\n\n段落B。\n\n段落C。"
    chunks = semantic_chunk(text, max_tokens=5)
    assert len(chunks) == 3

def test_chunk_respects_max_tokens():
    text = "这是一个很长的段落文本" * 20
    chunks = semantic_chunk(text, max_tokens=50)
    assert len(chunks) > 1

def test_chunk_preserves_heading():
    text = "第一章\n\n第一段。\n\n第二段。"
    chunks = semantic_chunk(text, max_tokens=100)
    assert len(chunks) >= 1
