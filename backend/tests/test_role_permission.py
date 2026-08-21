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


def test_migrate_visibility_on_legacy_db(tmp_path):
    """存量库（无 visibility 列）：init_db 补列并按关键词回填；二次执行不覆盖手动切换"""
    import sqlite3
    from sqlalchemy import text
    from app.database import init_db, get_engine, reset_engine

    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, filename TEXT, "
                 "file_type TEXT, file_size INTEGER, status TEXT, chunk_count INTEGER)")
    conn.execute("INSERT INTO documents (filename, file_type, file_size, status, chunk_count) "
                 "VALUES ('运营经理课件.docx','docx',1,'ready',0)")
    conn.execute("INSERT INTO documents (filename, file_type, file_size, status, chunk_count) "
                 "VALUES ('员工手册.docx','docx',1,'ready',0)")
    conn.commit()
    conn.close()

    try:
        init_db(str(db_path))
        engine = get_engine(str(db_path))
        with engine.connect() as c:
            rows = dict(c.execute(text("SELECT filename, visibility FROM documents")).fetchall())
        assert rows["运营经理课件.docx"] == "manager_only"
        assert rows["员工手册.docx"] == "all"

        # 经理手动改为 all 后，再次 init_db（模拟重启）不得覆盖
        with engine.begin() as c:
            c.execute(text("UPDATE documents SET visibility='all' "
                           "WHERE filename='运营经理课件.docx'"))
        init_db(str(db_path))
        with engine.connect() as c:
            vis = c.execute(text("SELECT visibility FROM documents "
                                 "WHERE filename='运营经理课件.docx'")).scalar()
        assert vis == "all"
    finally:
        reset_engine()


def test_api_role_permission():
    """API 级：X-Role 缺省/非法降级员工；员工列表过滤 + 写操作拒绝；经理全量可见"""
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from app.main import app
    from app.database import Base, get_db

    # 内存库需 StaticPool 共享单连接，否则每个新连接都是独立的空库
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    seed = Session()
    seed.add(Document(filename="员工手册.md", file_type="md", file_size=1, status="ready", visibility="all"))
    seed.add(Document(filename="运营经理课件.md", file_type="md", file_size=1, status="ready", visibility="manager_only"))
    seed.commit()
    seed.close()

    try:
        client = TestClient(app)
        # 员工：列表过滤经理专属，total 同步
        data = client.get("/api/documents", headers={"X-Role": "employee"}).json()["data"]
        assert data["total"] == 1
        assert all(d["visibility"] != "manager_only" for d in data["items"])
        # 经理：全量可见
        assert client.get("/api/documents", headers={"X-Role": "manager"}).json()["data"]["total"] == 2
        # 缺省/非法请求头降级员工
        assert client.get("/api/documents").json()["data"]["total"] == 1
        assert client.get("/api/documents", headers={"X-Role": "admin"}).json()["data"]["total"] == 1
        # 员工写操作被拒
        assert client.delete("/api/documents/1", headers={"X-Role": "employee"}).json()["code"] == "PERMISSION_DENIED"
        resp = client.patch("/api/documents/1/visibility", json={"visibility": "all"}, headers={"X-Role": "employee"})
        assert resp.json()["code"] == "PERMISSION_DENIED"
        # 经理可切换可见性
        resp = client.patch("/api/documents/1/visibility", json={"visibility": "manager_only"}, headers={"X-Role": "manager"})
        assert resp.json()["code"] == "SUCCESS"
    finally:
        app.dependency_overrides.clear()
