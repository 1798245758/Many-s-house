"""长期记忆管理 API 测试：列表 / 手动新增（key 派生幂等）/ 删除（走 InMemoryStore）"""
from fastapi.testclient import TestClient

from app.main import app
from app.services.retrieval.memory_store import get_memory_store, put_memory, read_memories

client = TestClient(app)


def test_list_empty():
    assert client.get("/api/memories").json()["data"] == []


def test_create_then_list():
    body = client.post("/api/memories",
                       json={"content": "回答要简洁", "category": "instruction"}).json()
    assert body["code"] == "SUCCESS"
    key = body["data"]["key"]
    assert key.startswith("m_")
    items = client.get("/api/memories").json()["data"]
    assert any(m["key"] == key and m["content"] == "回答要简洁"
               and m["category"] == "instruction" for m in items)


def test_create_is_idempotent_by_content():
    """同 content 两次新增派生同一 key，不产生重复条目"""
    client.post("/api/memories", json={"content": "用表格回答", "category": "format"})
    client.post("/api/memories", json={"content": "用表格回答", "category": "format"})
    matched = [m for m in read_memories(get_memory_store()) if m["content"] == "用表格回答"]
    assert len(matched) == 1


def test_delete_removes_item():
    put_memory(get_memory_store(), "tmp_key", "临时偏好", "preference")
    assert client.delete("/api/memories/tmp_key").json()["code"] == "SUCCESS"
    assert all(m["key"] != "tmp_key" for m in read_memories(get_memory_store()))


def test_delete_missing_key_still_ok():
    assert client.delete("/api/memories/not_exist").json()["code"] == "SUCCESS"
