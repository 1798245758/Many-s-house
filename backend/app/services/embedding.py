from sentence_transformers import SentenceTransformer
import struct
import numpy as np

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
    return _model

EMBEDDING_DIM = 768


def embed_text(text: str) -> bytes:
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return struct.pack(f"{EMBEDDING_DIM}f", *embedding.tolist())


def embed_text_list(texts: list[str]) -> list[bytes]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [struct.pack(f"{EMBEDDING_DIM}f", *emb.tolist()) for emb in embeddings]
