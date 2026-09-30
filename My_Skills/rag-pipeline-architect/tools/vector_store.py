"""ChromaDB 向量库抽象层

职责：向量的 CRUD + ANN 检索，作为项目中唯一的向量存取入口
使用方式：直接复制到项目 app/services/vector_store.py

关键设计：
- PersistentClient 嵌入式部署（无需启动服务）
- 进程级单例（双重检查锁）
- RLock 串行化所有操作（ChromaDB 非线程安全）
- ID 与 SQLite chunks.id 对齐（跨存储 JOIN 的基础）
- cosine 距离 → 转换为余弦相似度返回（1.0 - distance）
- reset_store() 供测试 fixture 切换临时目录
"""
import logging
import threading

logger = logging.getLogger(__name__)

_client = None
_collection = None
_lock = threading.Lock()
# ChromaDB 客户端非线程安全，所有读写操作需串行化
_op_lock = threading.RLock()


def get_collection():
    """获取 ChromaDB collection 单例（线程安全，双重检查锁）"""
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
            logger.info(f"ChromaDB collection已就绪: {CHROMA_COLLECTION}")
    return _collection


def reset_store():
    """重置单例（测试 fixture 切换临时目录时调用）"""
    global _client, _collection
    with _lock:
        _client = None
        _collection = None


def upsert_chunks(chunk_ids: list[int], embeddings: list[list[float]],
                  document_id: int, chunk_indices: list[int]) -> bool:
    """写入/更新向量（入库时调用）

    Args:
        chunk_ids: SQLite chunks 表主键列表
        embeddings: 对应的 768 维向量列表
        document_id: 所属文档 ID
        chunk_indices: 各 chunk 在文档内的序号
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
    """删除某文档的全部向量（文档删除/替换时调用）"""
    try:
        collection = get_collection()
        with _op_lock:
            collection.delete(where={"document_id": int(document_id)})
        return True
    except Exception as e:
        logger.error(f"按文档删除向量失败: {e}")
        return False


def delete_by_ids(chunk_ids: list[int]) -> bool:
    """按 chunk 主键删除向量"""
    if not chunk_ids:
        return True
    try:
        collection = get_collection()
        with _op_lock:
            collection.delete(ids=[str(cid) for cid in chunk_ids])
        return True
    except Exception as e:
        logger.error(f"按id删除向量失败: {e}")
        return False


def get_embeddings(chunk_ids: list[int]) -> dict[int, list[float]]:
    """批量读取已存向量（精排 rerank 时使用）

    Returns:
        dict: {chunk_id: vector}，缺失的 ID 不在字典中
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
    """ANN 向量相似度检索

    Args:
        query_vec: 768 维查询向量
        top_k: 返回条数

    Returns:
        list[(chunk_id, cosine_similarity)]，按相似度降序；失败返回空列表
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
        # cosine 距离 → 余弦相似度
        return [(int(cid), 1.0 - dist) for cid, dist in zip(ids, distances)]
    except Exception as e:
        logger.error(f"ChromaDB向量检索失败: {e}")
        return []


def count() -> int:
    """向量总数（健康检查用），失败返回 -1"""
    try:
        with _op_lock:
            return get_collection().count()
    except Exception as e:
        logger.error(f"ChromaDB count失败: {e}")
        return -1
