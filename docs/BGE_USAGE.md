# BGE模型使用说明

## 安装依赖
```bash
pip install -r requirements.txt
```

## 首次使用
首次使用时会自动下载BGE模型（约400MB），请确保网络连接。

## 配置
模型默认使用 `BAAI/bge-base-zh-v1.5`，如需更换模型，请修改 `app/services/embedding.py` 中的模型名称。

## API使用
```python
from app.services.embedding import embed_text, embed_text_list

# 单文本向量化
vector = embed_text("你的文本")

# 批量向量化
vectors = embed_text_list(["文本1", "文本2"])
```
