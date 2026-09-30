"""长期记忆提取链：每轮问答结束前用一次 LLM 调用，从「本轮问答对 + 既有记忆」
提炼确定性事实，输出结构化增删改操作，写入 SqliteStore。

与 rewrite_chain 同构：with_structured_output(method="function_calling")
（DeepSeek 不支持 json_schema）。失败降级为空操作，绝不阻断问答主链路。

build_memory_extractor 返回 extractor(question, answer, existing) -> MemoryUpdateResult。
apply_memory_ops(store, ops) 把操作落到 Store。
"""
import logging
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.services.retrieval.memory_store import put_memory, delete_memory

logger = logging.getLogger(__name__)

EXTRACT_PROMPT = (
    "你是企业问答助手的长期记忆管理器。从「本轮问答」中提炼值得跨会话恒定记住的"
    "确定性事实，并对照「既有记忆」决定增删改。只分析不回答。\n"
    "只提取恒定、跨轮有效的事实，分四类：\n"
    "  role（身份）：如「我是经理」「我在餐饮部」；\n"
    "  instruction（行为指令）：约束怎么做/做什么，如「回答要简洁」「找不到就说没有别编造」；\n"
    "  format（输出格式）：约束呈现形式，如「用表格回答」「分点列出」；\n"
    "  preference（偏好）：如「常问温暖基金」「偏好正式语气」。\n"
    "一次性/临时信息（如本次问题的具体答案内容）不记。\n"
    "对照既有记忆：语义重复→忽略；演变/冲突→update 或 delete（必须引用既有 key）；"
    "全新→create（新命名一个稳定短 key，小写字母/数字/下划线，如 user_role、answer_style）。\n"
    "无值得记录的事实→返回空 operations。\n\n"
    "既有记忆（key: [category] content）:\n{existing}\n\n"
    "本轮问题: {question}\n本轮回答片段: {answer}"
)


class MemoryOp(BaseModel):
    op: Literal["create", "update", "delete"] = Field(description="操作类型")
    key: str = Field(description="create 时新命名；update/delete 必须引用既有 key")
    content: str | None = Field(default=None, description="记忆内容，delete 时为空")
    category: Literal["role", "instruction", "format", "preference"] | None = Field(
        default=None, description="记忆类别")
    reason: str = Field(default="", description="决策依据（便于前端展示/调试）")


class MemoryUpdateResult(BaseModel):
    operations: list[MemoryOp] = Field(default=[])


def build_memory_extractor(api_key: str, model=None):
    """构建提取链：返回 extractor(question, answer, existing) -> MemoryUpdateResult

    existing：既有记忆列表（[{key, content, category}]）；model 仅供测试注入 mock。
    """
    if model is None:
        model = init_chat_model(
            "deepseek-chat",
            model_provider="openai",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0.0,
        )

    chain = (
        ChatPromptTemplate.from_template(EXTRACT_PROMPT)
        | model.with_structured_output(MemoryUpdateResult, method="function_calling")
    )

    def extractor(question: str, answer: str,
                  existing: list[dict] | None = None) -> MemoryUpdateResult:
        try:
            existing_text = "\n".join(
                f"{m['key']}: [{m['category']}] {m['content']}"
                for m in (existing or [])
            ) or "（无）"
            return chain.invoke({
                "question": question,
                "answer": answer or "（无）",
                "existing": existing_text,
            })
        except Exception as e:
            logger.warning(f"长期记忆提取失败，本轮不更新: {e}")
            return MemoryUpdateResult(operations=[])

    return extractor


def apply_memory_ops(store, ops: list[MemoryOp]):
    """把增删改操作落到 Store：
    create/update → put（update 指向不存在 key 时降级为 create，语义等价）；
    delete → delete（不存在静默跳过）。忽略缺 content 的非法 create/update。"""
    for op in ops or []:
        if op.op == "delete":
            delete_memory(store, op.key)
        elif op.op in ("create", "update"):
            if not op.content:
                continue
            put_memory(store, op.key, op.content, op.category or "preference")
