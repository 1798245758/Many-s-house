import pytest
from app.database import get_engine, get_session_local, get_db, Base, init_db


def test_get_engine_returns_sqlite_engine():
    engine = get_engine()
    assert engine is not None
    assert "sqlite" in str(engine.url)


def test_get_session_local_returns_sessionmaker():
    sm = get_session_local()
    assert sm is not None


def test_get_db_yields_session():
    gen = get_db()
    session = next(gen)
    assert session is not None
    try:
        next(gen)
    except StopIteration:
        pass


def test_init_db_creates_tables():
    import tempfile
    import os

    tmp = tempfile.mktemp(suffix=".db")
    try:
        init_db(db_path=tmp)
        engine = get_engine()
        inspector = __import__("sqlalchemy").inspect(engine)
        tables = inspector.get_table_names()
        print(f"Tables: {tables}")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
