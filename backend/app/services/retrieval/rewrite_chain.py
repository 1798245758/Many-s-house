"""Query 理解链：意图识别前的改写 + 分解 + 关键词优化（单次结构化调用）

一次 with_structured_output 调用同时产出：
  ① rewritten_question  口语转书面、补全省略成分后的规范问题
  ② sub_questions       复合问题分解出的子问题（0~3 个，简单问题为空）
  ③ keywords            覆盖原问题与全部子问题的检索关键词（2~6 个）

短期记忆消费点：可携带最近对话记录与上轮任务状态，用于指代消解与
澄清承接（上轮在等待澄清时，把本轮输入当作对上轮问题的补充）。
长期记忆消费点：可携带用户恒定事实（身份/指令/偏好），辅助 Query 理解。
失败降级为"原问题 + 空子问题 + 原问题作唯一关键词"，保证可用性优先。
build_query_rewriter 返回 rewriter(question, history, task_status, memories) -> RewriteResult。
"""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

logger = logging.getLogger(__name__)

REWRITE_PROMPT = (
    "你是企业知识库助手的查询理解器。对用户问题做三件事，只分析不回答：\n"
    "1. 改写 rewritten_question：口语转书面、消解指代、补全省略成分，"
    "得到意图不变的规范问题；问题已足够规范时原样保留；\n"
    "2. 分解 sub_questions：若是包含多个独立子问题的复合问句，拆成0~3个"
    "各自完整的子问题；单一问题留空列表；\n"
    "3. 关键词 keywords：提炼2~6个核心检索词（实体、制度名、业务术语等），"
    "需覆盖原问题与所有子问题的检索需求，去掉口语助词。\n"
    "最近对话记录（用于指代消解；若上轮任务状态为等待澄清，"
    "则本轮输入是对上轮问题的补充，改写时需合并上轮语境）:\n"
    "{history}\n"
    "上轮任务状态: {task_status}\n"
    "用户长期记忆（身份/指令/偏好，理解问题时可参考）:\n{memories}\n"
    "问题: {question}"
)


class RewriteResult(BaseModel):
    rewritten_question: str = Field(description="改写后的规范问题")
    sub_questions: list[str] = Field(default=[], description="分解出的子问题")
    keywords: list[str] = Field(default=[], description="优化后的检索关键词")
    degraded: SkipJsonSchema[bool] = Field(default=False, description="LLM 失败走兜底（内部标记，不进 LLM 工具 schema）")


def fallback_rewrite(question: str) -> RewriteResult:
    """降级兜底：原问题、无子问题、原问题作唯一关键词"""
    return RewriteResult(rewritten_question=question, sub_questions=[],
                         keywords=[question], degraded=True)


def build_query_rewriter(api_key: str, model=None):
    """构建 Query 理解链：返回
    rewriter(question, history=None, task_status="") -> RewriteResult

    history：最近对话条目列表（短期记忆）；task_status：上轮任务状态字符串；
    memories：已格式化的长期记忆文本段（可为空）。
    model 参数仅供测试注入 mock；缺省时初始化 DeepSeek（OpenAI 兼容协议）。
    """
    if model is None:
        model = init_chat_model(
            "deepseek-chat",
            model_provider="openai",
            base_url="https://api.deepseek.com",
            api_key=api_key,
            temperature=0.3,
        )

    # method="function_calling"：DeepSeek 不支持 json_schema 的 response_format，
    # 走工具调用协议实现结构化输出
    rewrite_chain = (
        ChatPromptTemplate.from_template(REWRITE_PROMPT)
        | model.with_structured_output(RewriteResult, method="function_calling")
    )

    def query_rewriter(question: str, history: list | None = None,
                       task_status: str = "", memories: str = "") -> RewriteResult:
        try:
            return rewrite_chain.invoke({
                "question": question,
                "history": "\n".join(history) if history else "（无）",
                "task_status": task_status or "（无）",
                "memories": memories or "（无）",
            })
        except Exception as e:
            logger.warning(f"Query改写失败，用原问题兜底: {e}")
            return fallback_rewrite(question)

    return query_rewriter
