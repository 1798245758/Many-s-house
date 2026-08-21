"""意图识别与信息完整性检查 Chain（两阶段 LLM 调用）

① classify_chain  分类：判断问题动作 refuse/clarify/direct/search + 风险理由
② slot_chain      槽位提取：任务/实体/时间/风险备注 + 检索关键词或澄清问题

普通函数 build_intent_gate 编排两阶段并按 action 分流，不依赖 LangGraph。
任一阶段失败均降级为"以原问题作关键词走检索"，保证可用性优先。
"""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

CLASSIFY_PROMPT = (
    "你是企业知识库助手的前置分析器。分析用户问题，选择唯一动作：\n"
    "refuse: 问题涉及违法、暴力、色情、涉密等有害或敏感内容；\n"
    "clarify: 问题意图模糊到无法判断用户在问什么，或缺失的关键信息会直接导致"
    "歧义、无法检索（采用宽松标准，一般性问题不要选此项）；\n"
    "direct: 闲聊、问候、常识类问题，无需企业知识库资料即可回答；\n"
    "search: 涉及企业管理事实（文化、制度、培训、薪酬福利、运营流程等），需要检索。\n"
    "若动作是 refuse，在 risk_reason 中给出简要理由；否则 risk_reason 留空字符串。\n"
    "问题: {question}"
)

SLOT_PROMPT = (
    "从用户问题中提取信息。动作类型: {action}\n"
    "task: 任务类型（如 制度咨询、流程咨询、闲聊，简短词语）；\n"
    "entities: 关键实体列表（机构、制度名、人员角色等），没有则空列表；\n"
    "time: 时间信息，没有则留空字符串；\n"
    "risk_note: 问题涉及的潜在风险备注，没有则留空字符串；\n"
    "search_keywords: 仅当动作为 search 时填写，由问题改写的2~6个核心检索关键词，"
    "否则留空列表；\n"
    "clarification_question: 仅当动作为 clarify 时填写，一句面向用户的澄清问题"
    "（指出缺失的信息并请用户补充），否则留空字符串。\n"
    "问题: {question}"
)


class ClassifyResult(BaseModel):
    action: str = Field(description="refuse|clarify|direct|search")
    risk_reason: str = ""


class SlotResult(BaseModel):
    task: str = ""
    entities: list[str] = []
    time: str = ""
    risk_note: str = ""
    search_keywords: list[str] = []
    clarification_question: str = ""


def build_intent_gate(api_key: str, model=None):
    """构建意图关卡：返回 gate(question) -> dict

    返回 dict 结构：{"action", "risk_reason", "slots"(SlotResult|None), "question"}
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

    classify_chain = (
        ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
        | model.with_structured_output(ClassifyResult)
    )
    slot_chain = (
        ChatPromptTemplate.from_template(SLOT_PROMPT)
        | model.with_structured_output(SlotResult)
    )

    def fallback_slots(question: str) -> SlotResult:
        return SlotResult(task="未知", search_keywords=[question])

    def intent_gate(question: str) -> dict:
        # 第一阶段：分类；失败降级为 search + 原问题关键词
        try:
            cls = classify_chain.invoke({"question": question})
        except Exception as e:
            logger.warning(f"意图分类失败，降级为检索: {e}")
            return {"action": "search", "risk_reason": "",
                    "slots": fallback_slots(question), "question": question}

        if cls.action == "refuse":
            return {"action": "refuse", "risk_reason": cls.risk_reason,
                    "slots": None, "question": question}

        # 第二阶段：槽位提取；失败用原问题兜底关键词
        try:
            slots = slot_chain.invoke({"question": question, "action": cls.action})
        except Exception as e:
            logger.warning(f"槽位提取失败，用原问题兜底: {e}")
            slots = fallback_slots(question)

        return {"action": cls.action, "risk_reason": cls.risk_reason,
                "slots": slots, "question": question}

    return intent_gate
