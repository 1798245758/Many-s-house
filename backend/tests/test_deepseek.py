import pytest
import struct
from unittest.mock import patch, MagicMock
from app.models.document import Document
from app.services.deepseek import DeepSeekClient
from app.services.ingestion.embedder import vectorize_chunks

def test_embed_returns_vector():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
    mock_resp.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=mock_resp):
        client = DeepSeekClient(api_key="sk-test")
        result = client.embed("test")
        assert result == [0.1, 0.2, 0.3]

def test_chat_returns_text():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": "AI回答"}}]}
    mock_resp.raise_for_status.return_value = None
    with patch("httpx.Client.post", return_value=mock_resp):
        client = DeepSeekClient(api_key="sk-test")
        answer = client.chat("ctx", "q")
        assert answer == "AI回答"

def test_client_raises_on_empty_key():
    with pytest.raises(ValueError, match="API Key"):
        DeepSeekClient(api_key="")

def test_vectorize_chunks():
    with patch.object(DeepSeekClient, "embed", return_value=[0.1, 0.2, 0.3]):
        client = DeepSeekClient(api_key="sk-test")
        results = vectorize_chunks(["text1", "text2"], client)
        assert len(results) == 2
        decoded = struct.unpack("3f", results[0])
        assert decoded == pytest.approx((0.1, 0.2, 0.3))

def test_save_chunks(db_session):
    from app.services.ingestion.indexer import save_chunks
    from app.models.chunk import Chunk

    doc = Document(filename="test.pdf", file_type="pdf", file_size=1000, status="pending")
    db_session.add(doc)
    db_session.commit()

    emb = struct.pack("3f", 0.1, 0.2, 0.3)
    save_chunks(db_session, doc.id, ["content1", "content2"], [emb, emb])

    assert db_session.query(Chunk).count() == 2
    updated_doc = db_session.query(Document).filter(Document.id == doc.id).first()
    assert updated_doc.chunk_count == 2
    assert updated_doc.status == "ready"
