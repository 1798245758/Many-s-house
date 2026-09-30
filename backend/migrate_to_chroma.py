"""一次性迁移脚本：将SQLite chunks.embedding中的存量JSON向量迁移到ChromaDB

用法（在backend目录下执行）:
    py migrate_to_chroma.py

特性:
    - 幂等：已存在于ChromaDB中的向量会跳过，可重复运行
    - 分批处理（默认500条/批），避免内存峰值
    - 结束后打印迁移统计与一致性校验结果
"""
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate_to_chroma")

BATCH_SIZE = 500
EXPECTED_DIM = 768


def migrate(db_path: str | None = None) -> dict:
    from app.database import get_engine, get_session_local, init_db
    from app.models.chunk import Chunk
    from app.services import vector_store

    init_db(db_path)
    db = get_session_local()()

    collection = vector_store.get_collection()

    total_scanned = 0
    total_migrated = 0
    total_skipped = 0
    total_failed = 0

    offset = 0
    while True:
        chunks = (
            db.query(Chunk)
            .filter(Chunk.embedding.isnot(None))
            .order_by(Chunk.id)
            .offset(offset)
            .limit(BATCH_SIZE)
            .all()
        )
        if not chunks:
            break
        total_scanned += len(chunks)

        # 查询本批中已存在的id，实现幂等
        batch_ids = [str(c.id) for c in chunks]
        existing = set(collection.get(ids=batch_ids, include=[])["ids"])

        new_ids, new_vecs, new_doc_ids, new_indices = [], [], [], []
        for c in chunks:
            if str(c.id) in existing:
                total_skipped += 1
                continue
            try:
                vec = json.loads(c.embedding)
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(f"解析向量失败，跳过 (chunk_id={c.id}): {e}")
                total_failed += 1
                continue
            if len(vec) != EXPECTED_DIM:
                logger.warning(
                    f"向量维度不匹配，跳过 (chunk_id={c.id}, dim={len(vec)}, 期望{EXPECTED_DIM})，"
                    f"请重新上传该文档以重新生成向量"
                )
                total_failed += 1
                continue
            new_ids.append(c.id)
            new_vecs.append(vec)
            new_doc_ids.append(c.document_id)
            new_indices.append(c.chunk_index)

        if new_ids:
            metadatas = [
                {"document_id": int(doc_id), "chunk_index": int(idx)}
                for doc_id, idx in zip(new_doc_ids, new_indices)
            ]
            collection.upsert(
                ids=[str(i) for i in new_ids],
                embeddings=new_vecs,
                metadatas=metadatas,
            )
            total_migrated += len(new_ids)
            logger.info(f"已迁移 {total_migrated} 条...")

        offset += BATCH_SIZE

    db.close()

    # 一致性校验：SQLite中有向量的chunk数 vs ChromaDB总数（仅供参考，
    # ChromaDB可能还包含新写入的、embedding列为空的chunk）
    sqlite_with_vec = total_scanned
    chroma_count = collection.count()
    logger.info("=" * 50)
    logger.info(f"迁移完成: 扫描 {total_scanned} 条, 新迁移 {total_migrated} 条, "
                f"已存在跳过 {total_skipped} 条, 解析失败 {total_failed} 条")
    logger.info(f"校验: SQLite含向量chunk数={sqlite_with_vec}, ChromaDB向量总数={chroma_count}")

    return {
        "scanned": total_scanned,
        "migrated": total_migrated,
        "skipped": total_skipped,
        "failed": total_failed,
        "chroma_count": chroma_count,
    }


if __name__ == "__main__":
    db_arg = sys.argv[1] if len(sys.argv) > 1 else None
    result = migrate(db_arg)
    if result["failed"] > 0:
        sys.exit(1)
