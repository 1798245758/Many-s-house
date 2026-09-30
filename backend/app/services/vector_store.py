"""ChromaDB向量库抽象层

封装ChromaDB嵌入式客户端（PersistentClient），作为项目中唯一的向量存取入口。
SQLite（chunks表）仍是chunk内容与元数据的唯一事实来源，
ChromaDB只存储向量及最小元数据（document_id、chunk_index），id与chunks.id对齐。

后续若更换向量库，只需替换本文件实现。
"""
import logging
import threading

logger = logging.getLogger(__name__)

_client = None
_collection = None
_lock = threading.Lock()
# ChromaDB客户端非线程安全，所有读写操作需串行化
_op_lock = threading.RLock()


def get_collection():
    """获取ChromaDB collection单例（线程安全）"""
    global _client, _collection
    if _collection is not None:
        return _collection
    with _lock:
        if _collection is None:
            import chromadb
            from app.config import CHROMA_DIR, CHROMA_COLLECTION

            CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
            _collection = _client.get_or_create_collection(
                name=CHROMA_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB collection已就绪: {CHROMA_COLLECTION} (path={CHROMA_DIR})")
    return _collection


def reset_store():
    """重置单例（主要用于测试切换临时目录）"""
    global _client, _collection
    with _lock:
        _client = None
        _collection = None


def upsert_chunks(chunk_ids: list[int], embeddings: list[list[float]], document_id: int, chunk_indices: list[int]) -> bool:
    """将chunk向量写入/更新到向量库

    Args:
        chunk_ids: chunks表主键列表
        embeddings: 对应的向量列表（768维）
        document_id: 所属文档id
        chunk_indices: 各chunk在文档内的序号

    Returns:
        bool: 是否写入成功
    """
    if not chunk_ids:
        return True
    try:
        collection = get_collection()
        with _op_lock:
            collection.upsert(
                ids=[str(cid) for cid in chunk_ids],
                embeddings=embeddings,
                metadatas=[
                    {"document_id": int(document_id), "chunk_index": int(idx)}
                    for idx in chunk_indices
                ],
            )
        return True
    except Exception as e:
        logger.error(f"向量写入ChromaDB失败 (document_id={document_id}): {e}")
        return False


def delete_by_document(document_id: int) -> bool:
    """删除某文档的全部向量"""
    try:
        collection = get_collection()
        with _op_lock:
            collection.delete(where={"document_id": int(document_id)})
        return True
    except Exception as e:
        logger.error(f"按文档删除ChromaDB向量失败 (document_id={document_id}): {e}")
        return False


def delete_by_ids(chunk_ids: list[int]) -> bool:
    """按chunk主键删除向量"""
    if not chunk_ids:
        return True
    try:
        collection = get_collection()
        with _op_lock:
            collection.delete(ids=[str(cid) for cid in chunk_ids])
        return True
    except Exception as e:
        logger.error(f"按id删除ChromaDB向量失败: {e}")
        return False


def get_embeddings(chunk_ids: list[int]) -> dict[int, list[float]]:
    """按chunk主键批量读取向量（用于检索后精排）

    Returns:
        dict: {chunk_id: 向量}，读取失败或id不存在时缺失对应键
    """
    if not chunk_ids:
        return {}
    try:
        collection = get_collection()
        with _op_lock:
            result = collection.get(
                ids=[str(cid) for cid in chunk_ids], include=["embeddings"]
            )
        ids = result.get("ids", [])
        embeddings = result.get("embeddings", [])
        return {int(cid): emb for cid, emb in zip(ids, embeddings)}
    except Exception as e:
        logger.error(f"ChromaDB读取向量失败: {e}")
        return {}


def query(query_vec: list[float], top_k: int = 10) -> list[tuple[int, float]]:
    """向量相似度检索

    Args:
        query_vec: 查询向量（768维）
        top_k: 返回条数

    Returns:
        list[(chunk_id, 余弦相似度)]，按相似度降序；失败时返回空列表
    """
    try:
        collection = get_collection()
        with _op_lock:
            total = collection.count()
        if total == 0:
            return []
        with _op_lock:
            result = collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, total),
            )
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        # cosine距离 -> 余弦相似度
        return [(int(cid), 1.0 - dist) for cid, dist in zip(ids, distances)]
    except Exception as e:
        logger.error(f"ChromaDB向量检索失败: {e}")
        return []


def count() -> int:
    """向量库中的向量总数（校验用），失败时返回-1"""
    try:
        with _op_lock:
            return get_collection().count()
    except Exception as e:
        logger.error(f"ChromaDB count失败: {e}")
        return -1
