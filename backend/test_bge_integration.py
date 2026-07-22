import pytest
from app.services.embedding import embed_text, embed_text_list, EMBEDDING_DIM

def test_embedding_dimension():
    """测试向量维度"""
    result = embed_text("测试文本")
    assert len(result) == EMBEDDING_DIM * 4  # 4 bytes per float

def test_embedding_list():
    """测试批量向量化"""
    texts = ["文本1", "文本2", "文本3"]
    results = embed_text_list(texts)
    assert len(results) == 3
    for result in results:
        assert len(result) == EMBEDDING_DIM * 4

def test_embedding_normalization():
    """测试向量归一化"""
    result = embed_text("测试文本")
    import struct
    vector = list(struct.unpack(f"{EMBEDDING_DIM}f", result))
    norm = sum(v * v for v in vector) ** 0.5
    assert abs(norm - 1.0) < 0.01  # 归一化向量的模应接近1
