from app.services.embedding import embed_text_list


def vectorize_chunks(texts: list[str], client=None) -> list[bytes]:
    """将文本块转换为向量"""
    return embed_text_list(texts)
