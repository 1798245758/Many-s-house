import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from app.database import Base

from app.models.document import Document
from app.models.chunk import Chunk
from app.models.query_history import QueryHistory
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
