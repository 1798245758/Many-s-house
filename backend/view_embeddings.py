#!/usr/bin/env python3
"""查看向量化结果"""

import sys
import os
import json
sys.path.append('.')

from app.database import get_engine, SessionLocal
from app.models.chunk import Chunk

def view_embeddings():
    """查看数据库中的向量化结果"""
    print("=" * 60)
    print("查看向量化结果")
    print("=" * 60)
    
    db = SessionLocal()
    try:
        # 查询所有chunks
        chunks = db.query(Chunk).order_by(Chunk.id.desc()).limit(5).all()
        
        if not chunks:
            print("\n⚠️  数据库中没有chunks，请先上传文件")
            return
        
        print(f"\n找到 {len(chunks)} 条记录（显示最新5条）\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"--- Chunk {i} (ID: {chunk.id}) ---")
            print(f"文档ID: {chunk.document_id}")
            print(f"内容: {chunk.content[:100]}...")
            
            if chunk.embedding:
                # 解析embedding
                try:
                    embedding = json.loads(chunk.embedding)
                    print(f"向量维度: {len(embedding)}")
                    print(f"向量前10个值: {[round(v, 4) for v in embedding[:10]]}")
                    print(f"向量范数: {sum(v*v for v in embedding) ** 0.5:.4f}")
                except Exception as e:
                    print(f"解析embedding失败: {e}")
            else:
                print("⚠️  无向量数据")
            
            print()
        
        print("=" * 60)
        
    finally:
        db.close()

if __name__ == "__main__":
    view_embeddings()