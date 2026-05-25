import hashlib
import struct

EMBEDDING_DIM = 384


def embed_text(text: str) -> bytes:
    result = [0.0] * EMBEDDING_DIM
    for i, char in enumerate(text):
        h = int(hashlib.md5(f"{i}:{char}".encode()).hexdigest(), 16)
        result[h % EMBEDDING_DIM] += 1.0
    norm = sum(v * v for v in result) ** 0.5
    if norm > 0:
        result = [v / norm for v in result]
    return struct.pack(f"{EMBEDDING_DIM}f", *result)


def embed_text_list(vectors: list[float]) -> list[float]:
    return vectors
