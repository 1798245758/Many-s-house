"""多跳推理链：基于已检索证据决定下一跳检索（单次结构化调用）

依赖链式多跳的核心决策点：每跳检索后由本链判断
  ① finished  证据已足以回答主问题 → 退出循环进入重排
  ② 继续      基于证据中已拿到的事实（如实体名）+ 剩余子问题，
              生成自包含的下一跳查询与关键词
典型场景：「张三公司的总部在哪个国家?」→ 先查「张三的公司」拿到公司名
→ 再查「该公司总部」才拿到答案。

失败降级（模块内完成）：剩余子问题非空时顺序消费队首作下一跳，
队列已空则直接终止循环，保证可用性优先。
build_hop_reasoner 返回 reasoner(question, evidence, remaining) -> HopDecision。
"""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

logger = logging.getLogger(__name__)

HOP_PROMPT = (
    "你是企业知识库助手的多跳检索推理器。根据主问题与已检索证据，"
    "判断是否需要继续检索：\n"
    "若证据已足以回答主问题，finished 置 true；\n"
    "否则 finished 置 false，并基于证据中已获得的事实（如已知实体名），"
    "结合剩余子问题，生成自包含、可直接检索的下一跳问题 next_query，"
    "以及2~6个检索关键词 next_keywords（不得包含指代词，需带上具体实体）。\n"
    "主问题: {question}\n"
    "已检索证据:\n{evidence}\n"
    "剩余子问题: {remaining}"
)


class HopDecision(BaseModel):
    finished: bool = Field(description="已检索证据是否足以回答主问题")
    next_query: str = Field(default="", description="下一跳完整检索问句（自包含）")
    next_keywords: list[str] = Field(default=[], description="下一跳检索关键词")
    degraded: SkipJsonSchema[bool] = Field(default=False, description="LLM 失败走兜底（内部标记，不进 LLM 工具 schema）")


def build_hop_reasoner(api_key: str, model=None):
    """构建多跳推理链：返回 reasoner(question, evidence, remaining) -> HopDecision

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
    hop_chain = (
        ChatPromptTemplate.from_template(HOP_PROMPT)
        | model.with_structured_output(HopDecision, method="function_calling")
    )

    def hop_reasoner(question: str, evidence: str, remaining: list) -> HopDecision:
        try:
            return hop_chain.invoke({
                "question": question,
                "evidence": evidence or "（暂无）",
                "remaining": "；".join(remaining) if remaining else "（无）",
            })
        except Exception as e:
            logger.warning(f"多跳推理失败，降级处理: {e}")
            if remaining:  # 顺序消费队首子问题作下一跳
                return HopDecision(finished=False, next_query=remaining[0],
                                   next_keywords=[remaining[0]], degraded=True)
            return HopDecision(finished=True, degraded=True)  # 队列已空，终止循环

    return hop_reasoner
