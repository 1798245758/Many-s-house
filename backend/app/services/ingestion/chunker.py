import re

def estimate_tokens(text: str) -> int:
    return len(text)

def semantic_chunk(text: str, max_tokens: int = 800) -> list[str]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks = []
    current = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if estimate_tokens(current + "\n\n" + para) <= max_tokens:
            current = para if not current else current + "\n\n" + para
        else:
            if current:
                chunks.append(current)
            while estimate_tokens(para) > max_tokens:
                split_idx = int(len(para) * max_tokens / estimate_tokens(para))
                chunks.append(para[:split_idx])
                para = para[split_idx:]
            current = para
    if current:
        chunks.append(current)
    return chunks
