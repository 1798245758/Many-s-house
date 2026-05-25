from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DB_PATH

Base = declarative_base()
_engine = None
_SessionLocal = None


def get_engine(db_path=None):
    global _engine
    if _engine is None:
        path = db_path or str(DB_PATH)
        _engine = create_engine(f"sqlite:///{path}", echo=False, connect_args={"check_same_thread": False})
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
