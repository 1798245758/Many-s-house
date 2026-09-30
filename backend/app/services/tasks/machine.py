"""任务状态机：唯一事实源，所有状态变更必须显式合法转移"""

QUEUED = "queued"        # 已创建，等待线程池工人
RUNNING = "running"      # 正在执行某阶段
BLOCKED = "blocked"      # 外部依赖等待（向量化服务退避重试窗口期）
SUCCEEDED = "succeeded"  # 终态：全部阶段完成
FAILED = "failed"        # 终态：阶段异常且重试耗尽
CANCELLED = "cancelled"  # 终态：用户取消

TERMINAL_STATES = {SUCCEEDED, FAILED, CANCELLED}

# 合法转移表（结合文档入库业务定义）
LEGAL_TRANSITIONS = {
    QUEUED: {RUNNING, CANCELLED},
    RUNNING: {BLOCKED, SUCCEEDED, FAILED, CANCELLED},
    BLOCKED: {RUNNING, FAILED, CANCELLED},
    SUCCEEDED: set(),
    FAILED: set(),
    CANCELLED: set(),
}


class IllegalTaskTransition(Exception):
    """非法状态转移"""


def transition(current: str, target: str) -> str:
    """校验并返回目标状态；同态保持视为无操作，非法转移抛异常"""
    if target == current:
        return target
    if target not in LEGAL_TRANSITIONS.get(current, set()):
        raise IllegalTaskTransition(f"非法任务状态转移: {current} -> {target}")
    return target
