"""意图识别与信息完整性检查图（LangGraph 两阶段状态图）

===== 1. 定义状态：在节点间流转的共享数据结构 =====
IntentState：question → classify 节点写入 action/risk_reason → 条件边分流
  ├─ refuse        → 直接 END（slots 保持 None）
  └─ 其余 action   → slot 节点提取槽位 → END

任一节点失败均降级为"以原问题作关键词走检索"，保证可用性优先。
build_intent_gate 返回 gate(question) -> dict 的调用入口，签名与旧版一致。
"""
import logging
from typing import Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
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
    # Literal 约束：LLM 输出非法取值时结构化解析失败，自动走降级路径
    action: Literal["refuse", "clarify", "direct", "search"] = Field(
        description="动作类型"
    )
    risk_reason: str = ""


class SlotResult(BaseModel):
    task: str = ""
    entities: list[str] = []
    time: str = ""
    risk_note: str = ""
    search_keywords: list[str] = []
    clarification_question: str = ""


class IntentState(TypedDict, total=False):
    """意图关卡状态：在 classify/slot 节点间流转"""
    question: str
    action: str
    risk_reason: str
    slots: SlotResult
    degraded: bool


def fallback_slots(question: str) -> SlotResult:
    """降级兜底槽位：task 未知、用原问题作检索关键词"""
    return SlotResult(task="未知", search_keywords=[question])


def build_intent_gate(api_key: str, model=None):
    """构建意图关卡状态图：返回 gate(question) -> dict

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

    # method="function_calling"：DeepSeek 不支持 json_schema 的 response_format，
    # 走工具调用协议实现结构化输出
    classify_chain = (
        ChatPromptTemplate.from_template(CLASSIFY_PROMPT)
        | model.with_structured_output(ClassifyResult, method="function_calling")
    )
    slot_chain = (
        ChatPromptTemplate.from_template(SLOT_PROMPT)
        | model.with_structured_output(SlotResult, method="function_calling")
    )

    # ===== 2. 定义节点：每个节点是一个函数，读 State、写 State =====
    def classify_node(state: IntentState) -> dict:
        """第一阶段：分类；失败降级为 search + 兜底槽位"""
        try:
            cls = classify_chain.invoke({"question": state["question"]})
            return {"action": cls.action, "risk_reason": cls.risk_reason}
        except Exception as e:
            logger.warning(f"意图分类失败，降级为检索: {e}")
            return {"action": "search", "risk_reason": "",
                    "slots": fallback_slots(state["question"]), "degraded": True}

    def slot_node(state: IntentState) -> dict:
        """第二阶段：槽位提取；失败用原问题兜底关键词"""
        if state.get("slots") is not None:  # classify 降级时已写入兜底槽位
            return {}
        try:
            slots = slot_chain.invoke({"question": state["question"],
                                       "action": state["action"]})
            return {"slots": slots}
        except Exception as e:
            logger.warning(f"槽位提取失败，用原问题兜底: {e}")
            return {"slots": fallback_slots(state["question"]), "degraded": True}

    # ===== 条件边：判断函数，入参为上一个节点写入后的状态 =====
    def route_classify(state: IntentState) -> str:
        return "end" if state.get("action") == "refuse" else "slot"

    # ===== 3. 构建图 =====
    graph = StateGraph(IntentState)
    graph.add_node("classify", classify_node)
    graph.add_node("slot", slot_node)

    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", route_classify, {"end": END, "slot": "slot"})
    graph.add_edge("slot", END)

    gate_graph = graph.compile()

    # ===== 4. 执行入口（对外签名与旧版一致）=====
    def intent_gate(question: str) -> dict:
        out = gate_graph.invoke({"question": question})
        return {
            "action": out["action"],
            "risk_reason": out.get("risk_reason", ""),
            "slots": out.get("slots"),  # refuse 或异常路径下可能为 None
            "question": question,
            "degraded": out.get("degraded", False),
        }

    return intent_gate
