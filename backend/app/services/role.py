"""角色权限：员工/经理双角色（无账户密码验证，前端自选）

角色通过 X-Role 请求头传递，缺省/非法值一律按 employee（最小权限原则）。
"""
from fastapi import Header
from sqlalchemy.orm import Session

from app.models.document import Document

ROLE_EMPLOYEE = "employee"
ROLE_MANAGER = "manager"
_VALID_ROLES = {ROLE_EMPLOYEE, ROLE_MANAGER}


def parse_role(value: str | None) -> str:
    """解析角色取值：仅 manager 视为经理，其余（含空/非法）均为员工"""
    return ROLE_MANAGER if value == ROLE_MANAGER else ROLE_EMPLOYEE


def get_role(x_role: str | None = Header(None, alias="X-Role")) -> str:
    """FastAPI 依赖：从 X-Role 请求头解析角色"""
    return parse_role(x_role)


def manager_doc_ids(db: Session) -> list[int]:
    """查询所有经理专属文档的 ID（员工检索时用于排除）"""
    rows = db.query(Document.id).filter(Document.visibility == "manager_only").all()
    return [r[0] for r in rows]
