from app.services.embedding import get_cached_embedding, embed_text_list


def vectorize_chunks(texts: list[str], client=None) -> list[bytes]:
    """将文本块批量转换为向量（利用BGE batch编码加速）"""
    if not texts:
        return []
    if len(texts) == 1:
        return [get_cached_embedding(texts[0])]
    return embed_text_list(texts)
