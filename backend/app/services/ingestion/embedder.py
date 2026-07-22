from app.services.embedding import get_cached_embedding


def vectorize_chunks(texts: list[str], client=None) -> list[bytes]:
    """将文本块转换为向量（带缓存）"""
    return [get_cached_embedding(t) for t in texts]
