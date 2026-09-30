"""长期记忆存储：LangGraph SqliteStore 进程级单例 + 全量读写 + 上限裁剪

存"确定性事实"（用户角色/指令/输出格式/偏好），跨会话、跨重启恒定生效。
每轮问答结束前由提取链增删改写入，下轮全量注入 rewrite/direct/answer 的 prompt。

与短期记忆的分工：短期（MemorySaver Checkpointer）存最近 3 轮问答，进程退出即清；
长期存恒定事实，落 SQLite 文件持久化。记忆仅作 prompt 语境，不参与权限判定。

关键陷阱（已实测）：连接必须 isolation_level=None（autocommit），否则 SqliteStore
内部 BEGIN 与 Python 隐式事务冲突，抛 cannot start a transaction within a transaction；
check_same_thread=False 兼容 FastAPI 同步多线程。
"""
import sqlite3
import logging
from datetime import datetime, timezone

from langgraph.store.sqlite import SqliteStore

from app.config import LTM_DB_PATH, MEMORY_MAX_ITEMS

logger = logging.getLogger(__name__)

NAMESPACE = ("memories",)          # 全局单命名空间（tuple）
_SEARCH_LIMIT = 1000               # 全量读用：Store.search 默认 limit=10，必须显式放大

_STORE = None                      # 进程级单例（与 _GRAPH_CACHE 同构，跨请求复用）


def get_memory_store():
    """懒加载进程级单例 SqliteStore（首次调用建连接并建表）"""
    global _STORE
    if _STORE is None:
        conn = sqlite3.connect(str(LTM_DB_PATH), check_same_thread=False,
                               isolation_level=None)
        store = SqliteStore(conn)
        store.setup()
        _STORE = store
    return _STORE


def set_memory_store(store):
    """测试注入口：替换为 InMemoryStore（零落盘、进程内），与短期记忆测试风格一致"""
    global _STORE
    _STORE = store


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_memories(store) -> list[dict]:
    """全量读命名空间，按 updated_at 升序返回 [{key, content, category, updated_at}]"""
    out = []
    for it in store.search(NAMESPACE, limit=_SEARCH_LIMIT):
        v = it.value or {}
        out.append({
            "key": it.key,
            "content": v.get("content", ""),
            "category": v.get("category", "preference"),
            "updated_at": v.get("updated_at", ""),
        })
    out.sort(key=lambda m: m["updated_at"])
    return out


def format_memories(memories: list[dict]) -> str:
    """记忆列表 → prompt 文本段，每行 `- [category] content`；空列表返回占位「（无）」"""
    if not memories:
        return "（无）"
    return "\n".join(f"- [{m['category']}] {m['content']}" for m in memories)


def put_memory(store, key: str, content: str, category: str = "preference"):
    """写入/覆盖一条记忆（纯数据 dict，可 msgpack 序列化），随后触发上限裁剪"""
    store.put(NAMESPACE, key,
              {"content": content, "category": category, "updated_at": _now()})
    _trim(store)


def delete_memory(store, key: str):
    """删除一条记忆（不存在则静默跳过）"""
    store.delete(NAMESPACE, key)


def _trim(store):
    """条目数超 MEMORY_MAX_ITEMS 时按 updated_at 删最旧"""
    memories = read_memories(store)
    overflow = len(memories) - MEMORY_MAX_ITEMS
    for m in memories[:max(overflow, 0)]:
        store.delete(NAMESPACE, m["key"])
