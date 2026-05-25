from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DB_PATH

Base = declarative_base()
_engine = None
_SessionLocal = None
_current_db_path = None


def reset_engine():
    global _engine, _SessionLocal, _current_db_path
    _engine = None
    _SessionLocal = None
    _current_db_path = None


def get_engine(db_path=None):
    global _engine, _current_db_path
    path = db_path or str(DB_PATH)
    if _engine is None or _current_db_path != path:
        _engine = create_engine(f"sqlite:///{path}", echo=False, connect_args={"check_same_thread": False})
        _current_db_path = path
    return _engine


def get_session_local():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def get_db():
    db = get_session_local()()
    try:
        yield db
    finally:
        db.close()


def init_db(db_path=None):
    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
