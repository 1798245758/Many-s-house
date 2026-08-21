from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker, declarative_base

from app.config import DB_PATH, MANAGER_KEYWORDS

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
    _migrate_documents_visibility(engine)


def _migrate_documents_visibility(engine):
    """存量库迁移：documents 补 visibility 列，并按文件名关键词回填经理专属

    create_all 不会给已有表补列，需手动 ALTER；列已存在则跳过。
    """
    inspector = inspect(engine)
    if not inspector.has_table("documents"):
        return
    columns = {c["name"] for c in inspector.get_columns("documents")}
    with engine.begin() as conn:
        if "visibility" not in columns:
            conn.execute(text(
                "ALTER TABLE documents ADD COLUMN visibility TEXT DEFAULT 'all'"
            ))
        # 回填：文件名含经理关键词的存量文档标为经理专属（幂等）
        for kw in MANAGER_KEYWORDS:
            conn.execute(text(
                "UPDATE documents SET visibility='manager_only' "
                "WHERE filename LIKE :pat AND visibility != 'manager_only'"
            ), {"pat": f"%{kw}%"})
