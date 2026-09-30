"""证据校验链：检索循环末端的四条件把关（单次结构化调用）

对精排后的证据做最终质量校验，四个条件全部满足才允许确定性作答：
  ① 关键问题（主问题与子问题）均有证据覆盖
  ② 来源满足权威性（正式制度/手册/培训资料为权威，来源不明存疑）
  ③ 引用能支持结论且相互不矛盾
  ④ 未发现权限和安全问题

不满足时输出缺失方面 + 针对同一问题换角度（同义词替换、上下位概念、
换资料类型视角）的补充查询，交由主图继续循环检索。
异常降级为"视为充分"放行，避免校验故障把循环卡死，保证可用性优先。
build_evidence_verifier 返回
  verifier(question, sub_questions, evidence) -> VerifyResult。
"""
import logging

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

logger = logging.getLogger(__name__)

VERIFY_PROMPT = (
    "你是企业知识库助手的证据校验器。按以下四个条件严格评估已检索证据：\n"
    "1. 关键问题全覆盖：主问题与每个子问题都能从证据中找到相关信息；\n"
    "2. 来源权威性：证据来自正式制度、员工手册、培训资料等权威资料，"
    "来源不明的内容视为存疑；\n"
    "3. 引用支持结论：证据内容能支撑对问题的回答，且相互之间不矛盾；\n"
    "4. 权限与安全：证据中不含当前角色不应获取的内容或敏感信息。\n"
    "四条件全部满足时 sufficient 置 true；\n"
    "否则 sufficient 置 false，在 missing_aspects 列出缺失或有疑点的方面，"
    "并针对同一问题换一个检索角度（同义词替换、上下位概念、换资料类型视角）"
    "生成补充问题 next_query 与2~6个补充关键词 next_keywords。\n"
    "主问题: {question}\n"
    "子问题清单: {sub_questions}\n"
    "已检索证据:\n{evidence}"
)


class VerifyResult(BaseModel):
    sufficient: bool = Field(description="四条件是否全部满足")
    missing_aspects: list[str] = Field(default=[], description="缺失或有疑点的方面")
    next_query: str = Field(default="", description="换角度的补充检索问句")
    next_keywords: list[str] = Field(default=[], description="补充检索关键词")
    degraded: SkipJsonSchema[bool] = Field(default=False, description="LLM 失败降级放行（内部标记，不进 LLM 工具 schema）")


def build_evidence_verifier(api_key: str, model=None):
    """构建证据校验链：返回 verifier(question, sub_questions, evidence) -> VerifyResult

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
    verify_chain = (
        ChatPromptTemplate.from_template(VERIFY_PROMPT)
        | model.with_structured_output(VerifyResult, method="function_calling")
    )

    def evidence_verifier(question: str, sub_questions: list, evidence: str) -> VerifyResult:
        try:
            return verify_chain.invoke({
                "question": question,
                "sub_questions": "；".join(sub_questions) if sub_questions else "（无）",
                "evidence": evidence or "（暂无）",
            })
        except Exception as e:
            logger.warning(f"证据校验失败，降级放行: {e}")
            return VerifyResult(sufficient=True, degraded=True)

    return evidence_verifier
