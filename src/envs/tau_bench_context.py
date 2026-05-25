"""
τ-bench Context 变量管理
用于在异步 rollout 中共享环境和状态
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

# Context vars 用于跨异步任务共享状态
CURRENT_TAU_ENV = contextvars.ContextVar("current_tau_env", default=None)
CURRENT_TAU_STATE = contextvars.ContextVar("current_tau_state", default=None)
CURRENT_ASSISTANT_CONTENT = contextvars.ContextVar("current_assistant_content", default="")


def make_initial_state(task_id: int = 0) -> dict:
    """创建初始状态"""
    return {
        "task_id": task_id,
        "total_reward": 0.0,
        "num_turns": 0,
        "num_tool_calls": 0,
        "action_history": [],
        "transferred_to_human": False,
        "done": False,
        "contaminated": False,
    }


def get_current_env() -> Any:
    """获取当前环境"""
    return CURRENT_TAU_ENV.get()


def get_current_state() -> dict:
    """获取当前状态"""
    state = CURRENT_TAU_STATE.get()
    if state is None:
        return make_initial_state()
    return state


def set_current_env(env: Any):
    """设置当前环境"""
    CURRENT_TAU_ENV.set(env)


def set_current_state(state: dict):
    """设置当前状态"""
    CURRENT_TAU_STATE.set(state)


def update_state_reward(reward: float):
    """更新状态奖励"""
    state = get_current_state()
    state["total_reward"] += reward


def add_action_to_history(tool: str, parameters: dict, is_error: bool = False, inc_reward: float = 0.0):
    """添加动作到历史"""
    state = get_current_state()
    import json

    action_record = {
        "tool": tool,
        "parameters": parameters,
        "param_str": json.dumps(parameters, sort_keys=True, ensure_ascii=False).lower(),
        "is_error": is_error,
        "inc_reward": inc_reward,
    }
    state["action_history"].append(action_record)


def increment_tool_calls():
    """增加工具调用计数"""
    state = get_current_state()
    state["num_tool_calls"] = state.get("num_tool_calls", 0) + 1


def increment_turns():
    """增加轮次计数"""
    state = get_current_state()
    state["num_turns"] = state.get("num_turns", 0) + 1


def mark_done():
    """标记完成"""
    state = get_current_state()
    state["done"] = True


def mark_transferred_to_human():
    """标记转移到人工客服"""
    state = get_current_state()
    state["transferred_to_human"] = True
