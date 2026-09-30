"""任务状态机测试：六态合法/非法转移"""
import pytest

from app.services.tasks.machine import (
    QUEUED, RUNNING, BLOCKED, SUCCEEDED, FAILED, CANCELLED,
    TERMINAL_STATES, transition, IllegalTaskTransition,
)


@pytest.mark.parametrize("current,target", [
    (QUEUED, RUNNING),
    (QUEUED, CANCELLED),
    (RUNNING, BLOCKED),
    (RUNNING, SUCCEEDED),
    (RUNNING, FAILED),
    (RUNNING, CANCELLED),
    (BLOCKED, RUNNING),
    (BLOCKED, FAILED),
    (BLOCKED, CANCELLED),
])
def test_legal_transitions(current, target):
    assert transition(current, target) == target


def test_same_state_is_noop():
    """同态保持（如 running→running）视为无操作，不抛异常"""
    assert transition(RUNNING, RUNNING) == RUNNING


@pytest.mark.parametrize("current,target", [
    (QUEUED, BLOCKED),
    (QUEUED, SUCCEEDED),
    (QUEUED, FAILED),
    (BLOCKED, SUCCEEDED),
    (SUCCEEDED, RUNNING),
    (FAILED, RUNNING),
    (CANCELLED, RUNNING),
])
def test_illegal_transitions_raise(current, target):
    with pytest.raises(IllegalTaskTransition):
        transition(current, target)


def test_terminal_states_have_no_outgoing():
    for t in TERMINAL_STATES:
        for target in (QUEUED, RUNNING, BLOCKED):
            with pytest.raises(IllegalTaskTransition):
                transition(t, target)
