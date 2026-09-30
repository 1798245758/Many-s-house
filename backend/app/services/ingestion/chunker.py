import re
from typing import List

def estimate_tokens(text: str) -> int:
    """估算token数量（简化版本，使用字符数近似）"""
    return len(text)


# 分隔符优先级：从粗到细逐级降级（段落 → 行 → 句子 → 子句标点），
# 均无法拆分时兜底字符硬切。
# 零宽断言模式切分后标点保留在前一段末尾，空白分隔符模式则直接丢弃。
SEPARATOR_PRIORITY = [
    r'\n\s*\n',                 # 段落（空行分隔）
    r'\n',                      # 单换行（行级）
    r'(?<=[。！？!?；;…])',      # 句子边界（句号类）
    r'(?<=[，、：:,.])',         # 子句标点（逗号顿号冒号）
]


def _split_recursive(text: str, max_tokens: int, level: int) -> List[str]:
    """按分隔符优先级递归切分：当前级切出的片段仍超限则降级到下一级，
    相邻片段贪心合并不超过 max_tokens"""
    if len(text) <= max_tokens:
        return [text]

    # 所有分隔符都试完：字符级硬切兜底，保证内容不丢、块不超限
    if level >= len(SEPARATOR_PRIORITY):
        return [text[i:i + max_tokens] for i in range(0, len(text), max_tokens)]

    parts = [p for p in re.split(SEPARATOR_PRIORITY[level], text) if p.strip()]
    if len(parts) <= 1:
        # 该级分隔符不存在（或切不出多段），降级尝试更细的分隔符
        return _split_recursive(text, max_tokens, level + 1)

    chunks = []
    buf = ""
    for part in parts:
        if len(part) > max_tokens:
            # 单片段超限：先落盘缓冲区，片段本身降级到下一级再切
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_split_recursive(part, max_tokens, level + 1))
            continue
        if buf and len(buf) + len(part) > max_tokens:
            chunks.append(buf)
            buf = part
        else:
            buf += part
    if buf:
        chunks.append(buf)
    return chunks


def _merge_tiny(chunks: List[str], min_tokens: int, max_tokens: int) -> List[str]:
    """合并过小的碎片块：优先并入前一块，尾部碎片再尝试回并，均不超上限"""
    if len(chunks) <= 1:
        return chunks
    merged: List[str] = []
    for c in chunks:
        if merged and len(merged[-1]) < min_tokens \
                and len(merged[-1]) + len(c) <= max_tokens:
            merged[-1] += c
        else:
            merged.append(c)
    if len(merged) > 1 and len(merged[-1]) < min_tokens \
            and len(merged[-2]) + len(merged[-1]) <= max_tokens:
        merged[-2] += merged[-1]
        merged.pop()
    return merged


def _apply_overlap(chunks: List[str], overlap_tokens: int, max_tokens: int) -> List[str]:
    """相邻块重叠：下一块开头携带上一块结尾的重叠文本；
    重叠会导致超限时放弃重叠，优先保证块不超 max_tokens"""
    result = [chunks[0]]
    for c in chunks[1:]:
        overlap = result[-1][-overlap_tokens:]
        if len(overlap) + len(c) <= max_tokens:
            result.append(overlap + c)
        else:
            result.append(c)
    return result


def semantic_chunk(
    text: str,
    min_tokens: int = 500,
    max_tokens: int = 1000,
    overlap_tokens: int = 100
) -> List[str]:
    """
    分级分隔符切块：按 段落 → 行 → 句子 → 子句标点 的优先级逐级降级，
    先用最粗的分隔符切，切出的片段仍超过 max_tokens 再用更细的分隔符，
    都不行则字符硬切兜底；随后合并过小碎片，并在相邻块间附加重叠。
    
    Args:
        text: 输入文本
        min_tokens: 最小token数（低于此值的碎片尝试与相邻块合并）
        max_tokens: 最大token数（块大小上限）
        overlap_tokens: 重叠token数，用于保持上下文连续性
        
    Returns:
        切分后的文本块列表
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # 如果文本足够短，直接返回
    if len(text) <= max_tokens:
        return [text]
    
    chunks = _split_recursive(text, max_tokens, level=0)
    chunks = _merge_tiny(chunks, min_tokens, max_tokens)
    if overlap_tokens > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap_tokens, max_tokens)
    return [c for c in chunks if c.strip()]

def split_text_with_overlap(
    text: str,
    chunk_size: int = 800,
    overlap_size: int = 100,
    min_chunk_size: int = 500
) -> List[str]:
    """
    基于固定大小的重叠切分（备选方案）
    
    Args:
        text: 输入文本
        chunk_size: 块大小
        overlap_size: 重叠大小
        min_chunk_size: 最小块大小
        
    Returns:
        切分后的文本块列表
    """
    if not text:
        return []
    
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 尝试在句子边界切分
        if end < len(text):
            # 向前寻找句子边界
            for punct in ['。', '！', '？', '.', '!', '?', '\n']:
                last_punct = text.rfind(punct, start + min_chunk_size, end)
                if last_punct != -1:
                    end = last_punct + 1
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # 下一块从重叠位置开始
        start = end - overlap_size
        if start >= len(text):
            break
    
    return chunks
