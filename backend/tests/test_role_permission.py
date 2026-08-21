"""角色权限测试：员工/经理双角色的解析、文档可见性与检索过滤"""
from app.models.chunk import Chunk
from app.models.document import Document
from app.routers.documents import detect_visibility
from app.services import vector_store
from app.services.retrieval.searcher import hybrid_search
from app.services.role import parse_role, manager_doc_ids


def _norm(vec: list[float]) -> list[float]:
    n = sum(v * v for v in vec) ** 0.5
    return [v / n for v in vec]


def _make_vec(seed: int) -> list[float]:
    v = [0.0] * 768
    v[seed % 768] = 1.0
    v[(seed * 7) % 768] = 0.5
    return _norm(v)


def test_parse_role():
    assert parse_role("manager") == "manager"
    assert parse_role("employee") == "employee"
    assert parse_role(None) == "employee"
    assert parse_role("admin") == "employee"
    assert parse_role("") == "employee"


def test_detect_visibility():
    assert detect_visibility("胖东来运营经理培训课件.docx") == "manager_only"
    assert detect_visibility("胖东来员工手册.docx") == "all"
    assert detect_visibility("") == "all"


def test_manager_doc_ids(db_session):
    d1 = Document(filename="普通.md", file_type="md", file_size=1, status="ready", visibility="all")
    d2 = Document(filename="经理资料.md", file_type="md", file_size=1, status="ready", visibility="manager_only")
    db_session.add_all([d1, d2])
    db_session.commit()
    assert manager_doc_ids(db_session) == [d2.id]


def _seed_two_docs(db_session):
    """造一个普通文档 + 一个经理专属文档，各一个 chunk 并入向量库"""
    doc_all = Document(filename="员工手册.md", file_type="md", file_size=1, status="ready", visibility="all")
    doc_mgr = Document(filename="运营经理课件.md", file_type="md", file_size=1, status="ready", visibility="manager_only")
    db_session.add_all([doc_all, doc_mgr])
    db_session.commit()
    c_all = Chunk(document_id=doc_all.id, chunk_index=0, content="员工休假制度", token_count=5)
    c_mgr = Chunk(document_id=doc_mgr.id, chunk_index=0, content="运营经理培训内容", token_count=5)
    db_session.add_all([c_all, c_mgr])
    db_session.commit()
    query_vec = _make_vec(77)
    assert vector_store.upsert_chunks([c_all.id, c_mgr.id], [query_vec, query_vec],
                                      document_id=doc_all.id, chunk_indices=[0, 0])
    # 上面向量库 document_id 仅用于按文档删除，检索过滤以 SQLite 的 chunk.document_id 为准
    return doc_all, doc_mgr, c_all, c_mgr, query_vec


def test_hybrid_search_excludes_manager_docs(db_session):
    _, doc_mgr, c_all, _, query_vec = _seed_two_docs(db_session)
    chunks = hybrid_search(db_session, query_vec, "培训", top_k=10,
                           exclude_doc_ids=[doc_mgr.id])
    ids = [c.id for c in chunks]
    assert c_all.id in ids
    assert all(c.document_id != doc_mgr.id for c in chunks)


def test_hybrid_search_no_exclusion_returns_all(db_session):
    _, _, c_all, c_mgr, query_vec = _seed_two_docs(db_session)
    chunks = hybrid_search(db_session, query_vec, "培训", top_k=10)
    ids = [c.id for c in chunks]
    assert c_all.id in ids and c_mgr.id in ids


def test_hybrid_search_fallback_excludes_manager_docs(db_session):
    """无命中兜底分支也必须排除经理专属文档（不入向量库，强制走兜底）"""
    doc_all = Document(filename="员工手册.md", file_type="md", file_size=1, status="ready", visibility="all")
    doc_mgr = Document(filename="运营经理课件.md", file_type="md", file_size=1, status="ready", visibility="manager_only")
    db_session.add_all([doc_all, doc_mgr])
    db_session.commit()
    c_mgr = Chunk(document_id=doc_mgr.id, chunk_index=0, content="运营经理培训内容", token_count=5)
    c_all = Chunk(document_id=doc_all.id, chunk_index=0, content="员工休假制度", token_count=5)
    db_session.add_all([c_mgr, c_all])
    db_session.commit()

    chunks = hybrid_search(db_session, _make_vec(500), "完全无关词xyz", top_k=5,
                           exclude_doc_ids=[doc_mgr.id])
    ids = [c.id for c in chunks]
    assert c_mgr.id not in ids
    assert c_all.id in ids
