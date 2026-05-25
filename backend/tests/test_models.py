import json
import pytest


def test_create_document(db_session):
    from app.models.document import Document

    doc = Document(filename="test.pdf", file_type="pdf", file_size=1024, status="pending")
    db_session.add(doc)
    db_session.commit()

    assert doc.id is not None


def test_create_chunk(db_session):
    from app.models.document import Document
    from app.models.chunk import Chunk

    doc = Document(filename="test.pdf", file_type="pdf", file_size=1024)
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(document_id=doc.id, chunk_index=0, content="Hello world", token_count=3)
    db_session.add(chunk)
    db_session.commit()

    assert chunk.id is not None
    assert chunk.document_id == doc.id
    assert chunk.document is not None


def test_chunk_cascade_delete(db_session):
    from app.models.document import Document
    from app.models.chunk import Chunk

    doc = Document(filename="test.pdf", file_type="pdf", file_size=1024)
    db_session.add(doc)
    db_session.commit()

    chunk = Chunk(document_id=doc.id, chunk_index=0, content="Hello world")
    db_session.add(chunk)
    db_session.commit()

    db_session.delete(doc)
    db_session.commit()

    remaining = db_session.query(Chunk).filter(Chunk.document_id == doc.id).all()
    assert len(remaining) == 0


def test_create_query_history(db_session):
    from app.models.query_history import QueryHistory

    sources = json.dumps(["doc1.pdf", "doc2.pdf"])
    qh = QueryHistory(query_text="What is AI?", answer_text="Artificial Intelligence", sources=sources)
    db_session.add(qh)
    db_session.commit()

    assert qh.id is not None
    assert qh.query_text == "What is AI?"
    assert qh.answer_text == "Artificial Intelligence"
    assert qh.sources == sources


def test_create_user(db_session):
    from app.models.user import User

    user = User(nickname="Alice", email="alice@example.com")
    db_session.add(user)
    db_session.commit()

    assert user.id is not None
    assert user.nickname == "Alice"
    assert user.email == "alice@example.com"


def test_create_setting(db_session):
    from app.models.setting import Setting

    setting = Setting(key="theme", value="dark")
    db_session.add(setting)
    db_session.commit()

    queried = db_session.query(Setting).filter(Setting.key == "theme").first()
    assert queried is not None
    assert queried.key == "theme"
    assert queried.value == "dark"
