from app.services.embedding import embed_text


def vectorize_chunks(texts: list[str], client=None) -> list[bytes]:
    return [embed_text(t) for t in texts]
