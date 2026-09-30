"""向量检索测试（基于ChromaDB）"""
import pytest
from app.models.chunk import Chunk
from app.models.document import Document
from app.services import vector_store
from app.services.retrieval.searcher import vector_search, hybrid_search


def _norm(vec: list[float]) -> list[float]:
    n = sum(v * v for v in vec) ** 0.5
    return [v / n for v in vec]


def _make_vec(seed: int) -> list[float]:
    """构造确定性的稀疏768维单位向量"""
    v = [0.0] * 768
    v[seed % 768] = 1.0
    v[(seed * 7) % 768] = 0.5
    return _norm(v)


def test_vector_search_empty(db_session):
    assert vector_search(db_session, _make_vec(1), top_k=5) == []


def test_vector_search_returns_nearest(db_session):
    query = _make_vec(3)
    near = _make_vec(3)
    far = _make_vec(500)
    assert vector_store.upsert_chunks([1, 2], [near, far], document_id=1, chunk_indices=[0, 1])

    results = vector_search(db_session, query, top_k=2)
    assert results[0][0] == 1
    assert results[0][1] == pytest.approx(1.0, abs=1e-3)


def test_hybrid_search_merges_keyword_and_vector(db_session):
    doc = Document(filename="t.md", file_type="md", file_size=1, status="ready")
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    c1 = Chunk(document_id=doc.id, chunk_index=0, content="TCP 三次握手建立连接", token_count=10)
    c2 = Chunk(document_id=doc.id, chunk_index=1, content="DNS 协议用于域名解析", token_count=10)
    db_session.add_all([c1, c2])
    db_session.commit()

    query_vec = _make_vec(9)
    assert vector_store.upsert_chunks(
        [c1.id, c2.id], [query_vec, _make_vec(700)], doc.id, [0, 1]
    )

    chunks = hybrid_search(db_session, query_vec, "TCP 三次握手", top_k=5)
    ids = [c.id for c in chunks]
    assert c1.id in ids


def test_hybrid_search_fallback_to_latest(db_session):
    """无任何命中时回退到最新chunk"""
    doc = Document(filename="t.md", file_type="md", file_size=1, status="ready")
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)
    c = Chunk(document_id=doc.id, chunk_index=0, content="不相关内容", token_count=5)
    db_session.add(c)
    db_session.commit()

    chunks = hybrid_search(db_session, _make_vec(42), "完全不匹配的关键词", top_k=5)
    assert [x.id for x in chunks] == [c.id]


def test_delete_by_document():
    assert vector_store.upsert_chunks([10, 11], [_make_vec(1), _make_vec(2)], 42, [0, 1])
    assert vector_store.count() == 2
    assert vector_store.delete_by_document(42)
    assert vector_store.count() == 0


def test_delete_by_ids():
    assert vector_store.upsert_chunks([20, 21], [_make_vec(3), _make_vec(4)], 43, [0, 1])
    assert vector_store.delete_by_ids([20])
    assert vector_store.count() == 1
