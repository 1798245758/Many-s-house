import pytest
from pathlib import Path
from app.services.ingestion.extractor import extract_text
from app.services.ingestion.cleaner import clean_text
from app.services.ingestion.chunker import semantic_chunk
from app.services.ingestion.pipeline import _is_image


def test_is_image_detects_png():
    assert _is_image("png") is True
    assert _is_image("jpg") is True
    assert _is_image("jpeg") is True
    assert _is_image("gif") is True


def test_is_image_rejects_text():
    assert _is_image("txt") is False
    assert _is_image("md") is False
    assert _is_image("pdf") is False

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


def test_chunk_paragraph_boundary_not_crossed_when_sentences_small():
    """段落优先：块上限够装单段时，跨段落合并不得混排两段的句子"""
    p1 = "第一条规定。" * 5    # 30 字符/段；两段合计 60 > 40 而每段 ≤ 40 → 应恰好两段两块；
    p2 = "第二条规定。" * 5    # 若按句全局混排，第二段首句会被并入块1，故用块1内容相等断言兜住
    chunks = semantic_chunk(p1 + "\n\n" + p2, max_tokens=40, overlap_tokens=0)
    assert len(chunks) == 2
    assert "第二条" not in chunks[0]
    assert chunks[0] == p1
    assert chunks[1] == p2


def test_chunk_oversized_paragraph_falls_back_to_sentence_split():
    """段落切出来太大 → 降级按句子切，句子不被拦腰截断"""
    para = "短句一。短句二。短句三。短句四。"  # 16 字符无段落分隔符，单句 4 字符 ≤ max
    chunks = semantic_chunk(para, max_tokens=15, overlap_tokens=0)
    assert len(chunks) == 2
    assert all(len(c) <= 15 for c in chunks)
    assert all(c.endswith("。") for c in chunks)
    assert "".join(chunks) == para


def test_chunk_no_punctuation_falls_back_to_clause_then_char():
    """无句号的长文本：先按子句标点（，）切，每个块带逗号边界而非硬切"""
    text = "甲条款内容，" * 4  # 24 字符全段无句号；按逗号切得 2×12 字符且前块以“，”结尾，字符硬切则得到 14+10 且无逗号边界
    chunks = semantic_chunk(text, max_tokens=14, overlap_tokens=0)
    assert len(chunks) == 2
    assert chunks[0].endswith("，")
    assert all(len(c) <= 14 for c in chunks)


def test_chunk_hard_cut_when_no_separator_available():
    """全无任何分隔符 → 字符级兜底硬切，内容无丢失"""
    text = "字" * 130  # 无标点无换行：任何一级分隔符都切不开，必须字符硬切 50+50+30，拼接可还原
    chunks = semantic_chunk(text, max_tokens=50, overlap_tokens=0)
    assert all(len(c) <= 50 for c in chunks)
    assert len(chunks) == 3
    assert "".join(chunks) == text


def test_chunk_overlap_prepended_to_next_chunk():
    """重叠：下一块开头携带上一块结尾的重叠文本，且块仍不超限"""
    p1 = "甲" * 30  # 第一段：重叠后块2 = 首块末10字符 + 第二段，长度 35 ≤ 40；
    p2 = "乙" * 25  # 若未做重叠则块2 长仅 25，前缀+长度联合断言可精确识别重叠生效
    chunks = semantic_chunk(p1 + "\n\n" + p2, max_tokens=40, overlap_tokens=10)
    assert len(chunks) == 2
    assert chunks[1].startswith("甲" * 10)
    assert chunks[1] == "甲" * 10 + p2
    assert all(len(c) <= 40 for c in chunks)


def test_extract_xmind(tmp_path):
    import zipfile, json
    f = tmp_path / "test.xmind"
    content = [{"id": "root", "title": "中心主题", "rootTopic": {
        "id": "r1", "title": "中心主题",
        "children": {"attached": [
            {"id": "c1", "title": "子主题A"},
            {"id": "c2", "title": "子主题B", "children": {"attached": [
                {"id": "c3", "title": "孙主题"}
            ]}}
        ]}
    }}]
    with zipfile.ZipFile(str(f), "w") as z:
        z.writestr("content.json", json.dumps(content, ensure_ascii=False))
    result = extract_text(f)
    assert "中心主题" in result
    assert "子主题A" in result
    assert "子主题B" in result
    assert "孙主题" in result
