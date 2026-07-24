#!/usr/bin/env python3
import sys
import os
sys.path.append('.')

# 设置环境变量
os.environ['USE_BGE_MODEL'] = 'false'

print("开始测试...")
sys.stdout.flush()

try:
    from app.services.embedding import embed_text, EMBEDDING_DIM
    print("导入成功")
    sys.stdout.flush()
    
    result = embed_text("测试文本")
    print(f"向量化成功，长度: {len(result)}")
    sys.stdout.flush()
    
    print(f"期望长度: {EMBEDDING_DIM * 4}")
    sys.stdout.flush()
    
    if len(result) == EMBEDDING_DIM * 4:
        print("测试通过！")
    else:
        print("测试失败！")
        
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
    sys.stdout.flush()