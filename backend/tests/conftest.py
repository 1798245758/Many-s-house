import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from app.database import Base

from app.models.document import Document
from app.models.chunk import Chunk
from app.models.query_history import QueryHistory
from app.models.query_trace import QueryTrace
from app.models.user import User
from app.models.setting import Setting


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    event.listen(engine, "connect", lambda conn, record: conn.execute("pragma foreign_keys=ON"))
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(bind=engine)
    session = TestingSession()
    yield session
    session.rollback()
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function", autouse=True)
def isolated_vector_store(tmp_path, monkeypatch):
    """将ChromaDB重定向到临时目录，避免测试污染真实向量库"""
    from app import config
    from app.services import vector_store

    monkeypatch.setattr(config, "CHROMA_DIR", tmp_path / "chroma")
    vector_store.reset_store()
    yield
    vector_store.reset_store()


@pytest.fixture(scope="function", autouse=True)
def isolated_memory_store():
    """长期记忆注入进程内 InMemoryStore（零落盘），每个用例独立，
    避免测试创建真实 long_term_memory.db 与跨用例记忆污染"""
    from langgraph.store.memory import InMemoryStore
    from app.services.retrieval import memory_store

    memory_store.set_memory_store(InMemoryStore())
    yield
    memory_store.set_memory_store(None)
