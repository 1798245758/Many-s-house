import struct
from app.services.deepseek import DeepSeekClient

def vectorize_chunks(texts: list[str], client: DeepSeekClient) -> list[bytes]:
    result = []
    for text in texts:
        vec = client.embed(text)
        buf = struct.pack(f"{len(vec)}f", *vec)
        result.append(buf)
    return result
