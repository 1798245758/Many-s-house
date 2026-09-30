"""Embedding 向量化模块（BGE 模型）

职责：文本 → 768维归一化向量
使用方式：直接复制到项目 app/services/embedding.py

关键设计：
- 单例模式 + 线程安全加载（首次调用耗时 10-30s）
- HF 镜像加速（国内环境必须）
- USE_BGE_MODEL=false 时降级为伪向量化（测试环境跳过模型加载）
- 返回 bytes（struct.pack）供 SQLite BLOB 存储
- embed_text_list 批量编码，batch_size=32 控制显存
"""
import os

# 必须在导入 sentence_transformers 之前设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')

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

# 向量维度（BAAI/bge-base-zh-v1.5 = 768）
EMBEDDING_DIM = 768


def get_model():
    """获取 BGE 模型单例（线程安全，带超时保护）"""
    global _model, _model_loading

    if _model is not None:
        return _model

    if _model_loading:
        start_time = time.time()
        while _model_loading and _model is None:
            time.sleep(0.1)
            if time.time() - start_time > 30:
                raise TimeoutError("模型加载超时")
        return _model

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

            def load_model():
                global _model, _model_loading
                try:
                    _model = SentenceTransformer(BGE_MODEL_NAME)
                    logger.info(f"BGE模型加载成功，耗时: {time.time() - start_time:.2f}秒")
                except Exception as e:
                    logger.error(f"加载BGE模型失败: {e}")
                    _model = None
                finally:
                    _model_loading = False

            load_thread = threading.Thread(target=load_model)
            load_thread.daemon = True
            load_thread.start()
            load_thread.join(timeout=BGE_MODEL_TIMEOUT)

            if _model is None:
                raise TimeoutError(f"BGE模型加载超时（{BGE_MODEL_TIMEOUT}秒）")
            return _model

        except Exception as e:
            _model_loading = False
            logger.error(f"加载BGE模型失败: {e}")
            raise


def pseudo_embed_text(text: str) -> bytes:
    """伪向量化（基于 MD5 哈希，仅用于测试环境）"""
    result = [0.0] * EMBEDDING_DIM
    for i, char in enumerate(text):
        h = int(hashlib.md5(f"{i}:{char}".encode()).hexdigest(), 16)
        result[h % EMBEDDING_DIM] += 1.0
    norm = sum(v * v for v in result) ** 0.5
    if norm > 0:
        result = [v / norm for v in result]
    return struct.pack(f"{EMBEDDING_DIM}f", *result)


def embed_text(text: str) -> bytes:
    """文本 → 768维向量（bytes 格式，供 SQLite BLOB 存储）

    Returns:
        bytes: struct.pack 的 768 个 float32
    """
    try:
        model = get_model()
        if model is None:
            return pseudo_embed_text(text)
        embedding = model.encode(text, normalize_embeddings=True)
        return struct.pack(f"{EMBEDDING_DIM}f", *embedding.tolist())
    except Exception as e:
        logger.error(f"文本向量化失败: {e}")
        return pseudo_embed_text(text)


@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> bytes:
    """带 LRU 缓存的向量化（重复查询免二次编码）"""
    return embed_text(text)


def embed_text_list(texts: list[str]) -> list[bytes]:
    """批量向量化（入库时使用，batch_size=32 控制显存）"""
    try:
        model = get_model()
        if model is None:
            return [pseudo_embed_text(t) for t in texts]
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
        return [struct.pack(f"{EMBEDDING_DIM}f", *emb.tolist()) for emb in embeddings]
    except Exception as e:
        logger.error(f"批量文本向量化失败: {e}")
        return [pseudo_embed_text(t) for t in texts]
