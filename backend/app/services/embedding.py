from sentence_transformers import SentenceTransformer
from functools import lru_cache
import struct
import logging
import hashlib
import threading
import time

logger = logging.getLogger(__name__)

_model = None
_model_loading = False
_model_lock = threading.Lock()

def get_model():
    """获取BGE模型实例（单例模式，带超时）"""
    global _model, _model_loading
    
    # 如果模型已加载，直接返回
    if _model is not None:
        return _model
    
    # 如果模型正在加载，等待
    if _model_loading:
        logger.info("模型正在加载中，等待...")
        start_time = time.time()
        while _model_loading and _model is None:
            time.sleep(0.1)
            if time.time() - start_time > 30:  # 30秒超时
                logger.error("等待模型加载超时")
                raise TimeoutError("模型加载超时")
        return _model
    
    # 尝试加载模型
    with _model_lock:
        if _model is not None:
            return _model
        
        _model_loading = True
        try:
            from app.config import USE_BGE_MODEL, BGE_MODEL_NAME, BGE_MODEL_TIMEOUT
            
            if not USE_BGE_MODEL:
                logger.info("BGE模型已禁用，使用伪向量化")
                _model_loading = False
                return None
            
            logger.info(f"开始加载BGE模型: {BGE_MODEL_NAME}")
            start_time = time.time()
            
            # 在后台线程中加载模型
            def load_model():
                global _model
                try:
                    _model = SentenceTransformer(BGE_MODEL_NAME)
                    logger.info(f"BGE模型加载成功，耗时: {time.time() - start_time:.2f}秒")
                except Exception as e:
                    logger.error(f"加载BGE模型失败: {e}")
                    _model = None
                finally:
                    global _model_loading
                    _model_loading = False
            
            load_thread = threading.Thread(target=load_model)
            load_thread.daemon = True
            load_thread.start()
            
            # 等待模型加载完成
            load_thread.join(timeout=BGE_MODEL_TIMEOUT)
            
            if _model is None:
                logger.error(f"BGE模型加载超时（{BGE_MODEL_TIMEOUT}秒）")
                raise TimeoutError(f"BGE模型加载超时（{BGE_MODEL_TIMEOUT}秒）")
            
            return _model
            
        except Exception as e:
            _model_loading = False
            logger.error(f"加载BGE模型失败: {e}")
            raise

# 伪向量化函数（当BGE模型不可用时使用）
EMBEDDING_DIM = 768

def pseudo_embed_text(text: str) -> bytes:
    """伪向量化（基于MD5哈希）"""
    result = [0.0] * EMBEDDING_DIM
    for i, char in enumerate(text):
        h = int(hashlib.md5(f"{i}:{char}".encode()).hexdigest(), 16)
        result[h % EMBEDDING_DIM] += 1.0
    norm = sum(v * v for v in result) ** 0.5
    if norm > 0:
        result = [v / norm for v in result]
    return struct.pack(f"{EMBEDDING_DIM}f", *result)


def embed_text(text: str) -> bytes:
    """将文本转换为向量
    
    Args:
        text: 输入文本
        
    Returns:
        bytes: 768维向量的字节表示
    """
    try:
        model = get_model()
        if model is None:
            # 使用伪向量化
            logger.debug("使用伪向量化")
            return pseudo_embed_text(text)
        
        embedding = model.encode(text, normalize_embeddings=True)
        return struct.pack(f"{EMBEDDING_DIM}f", *embedding.tolist())
    except Exception as e:
        logger.error(f"文本向量化失败: {e}")
        # 返回伪向量作为后备
        return pseudo_embed_text(text)


@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> bytes:
    """缓存文本的embedding结果"""
    return embed_text(text)


def embed_text_list(texts: list[str]) -> list[bytes]:
    """批量将文本转换为向量
    
    Args:
        texts: 输入文本列表
        
    Returns:
        list[bytes]: 向量字节表示列表
    """
    try:
        model = get_model()
        if model is None:
            # 使用伪向量化
            logger.debug("批量使用伪向量化")
            return [pseudo_embed_text(t) for t in texts]
        
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [struct.pack(f"{EMBEDDING_DIM}f", *emb.tolist()) for emb in embeddings]
    except Exception as e:
        logger.error(f"批量文本向量化失败: {e}")
        # 返回伪向量列表作为后备
        return [pseudo_embed_text(t) for t in texts]
