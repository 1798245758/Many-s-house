from sentence_transformers import SentenceTransformer
import struct
import logging

logger = logging.getLogger(__name__)

_model = None

def get_model():
    """获取BGE模型实例（单例模式）"""
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer('BAAI/bge-base-zh-v1.5')
        except Exception as e:
            logger.error(f"加载BGE模型失败: {e}")
            raise
    return _model

EMBEDDING_DIM = 768


def embed_text(text: str) -> bytes:
    """将文本转换为BGE向量
    
    Args:
        text: 输入文本
        
    Returns:
        bytes: 768维向量的字节表示
    """
    try:
        model = get_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return struct.pack(f"{EMBEDDING_DIM}f", *embedding.tolist())
    except Exception as e:
        logger.error(f"文本向量化失败: {e}")
        # 返回零向量作为后备
        return struct.pack(f"{EMBEDDING_DIM}f", *[0.0] * EMBEDDING_DIM)


def embed_text_list(texts: list[str]) -> list[bytes]:
    """批量将文本转换为BGE向量
    
    Args:
        texts: 输入文本列表
        
    Returns:
        list[bytes]: 向量字节表示列表
    """
    try:
        model = get_model()
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [struct.pack(f"{EMBEDDING_DIM}f", *emb.tolist()) for emb in embeddings]
    except Exception as e:
        logger.error(f"批量文本向量化失败: {e}")
        # 返回零向量列表作为后备
        return [struct.pack(f"{EMBEDDING_DIM}f", *[0.0] * EMBEDDING_DIM) for _ in texts]
