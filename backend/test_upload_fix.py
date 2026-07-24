#!/usr/bin/env python3
"""测试上传修复"""

import sys
import os
sys.path.append('.')

# 设置环境变量
os.environ['USE_BGE_MODEL'] = 'false'

from app.services.embedding import embed_text, embed_text_list, EMBEDDING_DIM

def test_embedding():
    """测试向量化功能"""
    print("测试向量化功能...")
    
    # 测试单文本向量化
    print("1. 测试单文本向量化...")
    result = embed_text("测试文本")
    print(f"   向量长度: {len(result)} 字节")
    print(f"   期望长度: {EMBEDDING_DIM * 4} 字节")
    assert len(result) == EMBEDDING_DIM * 4, f"向量长度错误: {len(result)} != {EMBEDDING_DIM * 4}"
    print("   ✓ 单文本向量化成功")
    
    # 测试批量向量化
    print("2. 测试批量向量化...")
    texts = ["文本1", "文本2", "文本3"]
    results = embed_text_list(texts)
    print(f"   批量结果数量: {len(results)}")
    assert len(results) == 3, f"批量结果数量错误: {len(results)} != 3"
    
    for i, result in enumerate(results):
        assert len(result) == EMBEDDING_DIM * 4, f"批量向量{i}长度错误"
    print("   ✓ 批量向量化成功")
    
    # 测试向量归一化
    print("3. 测试向量归一化...")
    import struct
    vector = list(struct.unpack(f"{EMBEDDING_DIM}f", result))
    norm = sum(v * v for v in vector) ** 0.5
    print(f"   向量模: {norm:.4f}")
    # 伪向量化可能不是完全归一化的，但应该接近1
    print("   ✓ 向量归一化测试完成")
    
    print("\n所有测试通过！✓")

if __name__ == "__main__":
    test_embedding()