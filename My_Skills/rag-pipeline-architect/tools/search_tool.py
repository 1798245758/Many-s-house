"""LangChain @tool 检索工具封装

职责：将 hybrid_search 封装为 LangChain Tool，供 Agent 图中 search_node 调用
使用方式：直接复制到项目 app/services/retrieval/search_tool.py

=== 工具描述遵循 5 要素规范 ===
1. 功能定义：一句话说清能力
2. 正向触发条件：何时该调用
3. 负向触发条件：何时不该调用（最关键）
4. 参数使用规范：query 如何填写
5. 返回值语义与异常处理：空结果如何应对

=== 核心设计约束 ===
- 向量在工具内部生成（embed_query），绝不作为参数暴露给 LLM
- LLM 只需传入文本 query，工具内部完成：文本→向量→混合检索→返回结果
- 参数保持极简（只有 query: str），降低 LLM 调用错误率
"""
import struct
import logging
from langchain_core.tools import tool
from sqlalchemy.orm import Session

from app.services.embedding import embed_text, EMBEDDING_DIM
from app.services.retrieval.searcher import hybrid_search
from app.services.role import ROLE_MANAGER, manager_doc_ids

logger = logging.getLogger(__name__)


def embed_query(text: str) -> list[float]:
    """文本 → 768 维浮点向量（工具内部使用，不暴露给 LLM）"""
    packed = embed_text(text)
    return list(struct.unpack(f"{EMBEDDING_DIM}f", packed))


def create_search_tool(db: Session, role: str = "employee"):
    """工厂函数：创建绑定了 db session 和 role 的检索工具实例

    为什么用工厂而非全局 @tool：
    - db 是请求级资源，不能在全局工具中闭包捕获
    - role 决定权限过滤范围，每请求不同
    - Agent 图节点调用时：tool = create_search_tool(db, role); tool.invoke(query)

    Args:
        db: 请求级 SQLAlchemy Session
        role: 用户角色（employee/manager）

    Returns:
        LangChain Tool 实例
    """

    @tool
    def search_knowledge_base(query: str) -> dict:
        """在企业知识库中检索与问题相关的资料片段。

        知识库内容为企业内部管理资料，涵盖：企业文化、员工手册、管理制度、
        培训体系、薪酬福利、运营流程、服务标准等主题。

        何时调用：当问题涉及上述企业管理主题，需要事实性资料支撑回答时。
        何时不调用：闲聊问候、通用常识、与企业管理无关的问题，直接回答即可，
        不要调用本工具。问题含糊但疑似涉及企业管理时，优先调用检索。

        参数 query：必须为提炼后的 2~6 个核心检索关键词（实体名、制度名、
        业务术语），不要原样照搬用户完整问题。
        正确示例："员工 温暖基金 发放标准"
        错误示例："我想问一下公司的温暖基金是怎么发放的呀？"

        返回：dict 格式 {"results": [...], "count": int}
        - results 为资料片段列表，每项含 content（内容）和 source（来源文档名）
        - count=0 表示知识库中无相关内容，此时必须诚实告知用户
          "根据现有资料无法回答此问题"，严禁编造答案。
        """
        try:
            # 向量在工具内部生成，不暴露给 LLM
            query_vec = embed_query(query)

            # 角色权限过滤
            exclude = manager_doc_ids(db) if role != ROLE_MANAGER else None

            # 混合检索
            chunks = hybrid_search(db, query_vec, query, top_k=10,
                                   exclude_doc_ids=exclude)

            results = [
                {
                    "id": c.id,
                    "content": c.content,
                    "source": c.document.filename if c.document else "未知来源",
                }
                for c in chunks
            ]
            return {"results": results, "count": len(results)}

        except Exception as e:
            logger.error(f"检索工具执行失败: {e}")
            return {"results": [], "count": 0, "error": str(e)}

    return search_knowledge_base


# ===== 在 Agent 图节点中的标准调用方式 =====
#
# def search_node(state: RAGState, config: RunnableConfig) -> dict:
#     """图中检索节点：从 config 取 db，调用 hybrid_search，立即转纯数据 dict"""
#     db = config["configurable"]["db"]
#     role = state.get("role", "employee")
#
#     # 方式 A：直接调用底层函数（推荐，性能最优）
#     exclude = manager_doc_ids(db) if role != ROLE_MANAGER else None
#     hits = hybrid_search(db, state["embedding"], state["current_query"],
#                          top_k=HOP_TOP_K_PER_HOP, exclude_doc_ids=exclude)
#
#     # 方式 B：通过 @tool 封装调用（适合需要工具描述的场景）
#     # search_tool = create_search_tool(db, role)
#     # result = search_tool.invoke(state["current_query"])
#     # hits = result["results"]
#
#     # 关键：ORM → 纯数据 dict（state 必须可 msgpack 序列化）
#     seen = {c["id"] for c in state.get("all_hits", [])}
#     new_hits = [{"id": c.id, "content": c.content,
#                  "document_name": c.document.filename if c.document else "未知"}
#                 for c in hits if c.id not in seen]
#
#     return {"all_hits": state.get("all_hits", []) + new_hits,
#             "hop_count": state.get("hop_count", 0) + 1,
#             "no_new_hits": not new_hits}
