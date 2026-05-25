"""
TauBenchInteraction: IG-GRPO 的 veRL BaseInteraction 实现
集成 JIG (Joint Information Gain) 奖励计算到 veRL 训练流程
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Optional

from verl.interactions.base import BaseInteraction

from .tau_bench_context import (
    CURRENT_TAU_ENV,
    CURRENT_TAU_STATE,
    CURRENT_ASSISTANT_CONTENT,
    make_initial_state,
)
from .jig_components import (
    JointInformationGain,
    SustainedExplorationBonus,
    CurriculumScheduler,
)


logger = logging.getLogger(__name__)


# 与原项目保持一致的禁止 token
FORBIDDEN_TEMPLATE_TOKENS = ["...", "```", "<strip"]


def _has_forbidden_token(content: str) -> bool:
    if not content:
        return False
    return any(tok in content for tok in FORBIDDEN_TEMPLATE_TOKENS)


def _extract_latest_assistant_content(messages: list[dict]) -> str:
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def _param_str(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False).lower()


def _compute_binary_reward(state: dict) -> float:
    """二元奖励：outcome >= 1.0 → 1, 否则 0"""
    return 1.0 if state["total_reward"] >= 1.0 else 0.0


def _compute_prm_lite_reward(state: dict) -> float:
    """PRM-Lite 奖励（简化版）"""
    outcome = 1.0 if state["total_reward"] >= 1.0 else 0.0
    history = state.get("action_history", [])

    # 简化的过程分数
    if not history:
        return outcome

    # 基于工具调用多样性
    tool_counts = {}
    for action in history:
        tool = action.get("tool", "")
        if tool not in ("think", "implicit_think"):
            tool_counts[tool] = tool_counts.get(tool, 0) + 1

    if tool_counts:
        diversity = len(tool_counts) / len(history)
        process_score = min(diversity, 1.0)
    else:
        process_score = 0.0

    return outcome + 0.3 * process_score


def _compute_jig_reward(state: dict, jig_computer: JointInformationGain, sustained_bonus: SustainedExplorationBonus) -> float:
    """IG-GRPO: outcome + curriculum_alpha * (JIG + sustained_bonus)"""
    outcome = 1.0 if state["total_reward"] >= 1.0 else 0.0

    history = state.get("action_history", [])
    if not history:
        return outcome

    # 计算 JIG
    jig_scores = []
    prev_tool = None

    for action in history:
        tool = action.get("tool", "")
        if tool in ("think", "implicit_think"):
            continue

        params = action.get("parameters", {})
        state_repr = _param_str(params)

        jig = jig_computer.compute_jig(state_repr, tool, prev_tool)
        jig_scores.append(jig)

        prev_tool = tool

    avg_jig = sum(jig_scores) / len(jig_scores) if jig_scores else 0.0

    # 更新探索持续性
    if sustained_bonus:
        sustained_bonus.update(avg_jig)
        sustained_value = sustained_bonus.compute_bonus()
    else:
        sustained_value = 0.0

    # 课程学习权重
    curriculum_alpha = state.get("curriculum_alpha", 0.3)

    total_reward = outcome + curriculum_alpha * (avg_jig + sustained_value)

    return max(0.0, min(2.0, total_reward))


_REWARD_FUNCTIONS = {
    "binary": _compute_binary_reward,
    "prm_lite": _compute_prm_lite_reward,
    # JIG 在 TauBenchInteractionIG 类中特殊处理
}


class TauBenchInteractionIG(BaseInteraction):
    """
    τ-bench 环境交互类（IG-GRPO 版本）
    负责管理 rollout 过程中的环境和状态，添加 JIG 奖励计算
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.env_name: str = config.get("env_name", "retail")
        self.user_strategy: str = config.get("user_strategy", "llm")
        self.user_model: str = config.get(
            "user_model", "Qwen/Qwen2.5-72B-Instruct-AWQ"
        )
        self.user_provider: str = config.get("user_provider", "openai")
        self.user_base_url: str = config.get(
            "user_base_url", "http://localhost:8001/v1"
        )
        self.task_split: str = config.get("task_split", "train")
        self.max_turns: int = int(config.get("max_turns", 30))

        # 奖励模式
        self.reward_mode: str = config.get("reward_mode", "binary")

        # IG-GRPO: JIG 配置
        jig_config = config.get("jig_config", {})
        self.use_jig = self.reward_mode == "jig" or jig_config.get("enabled", False)

        if self.use_jig:
            # 初始化 JIG 组件
            alpha_state = jig_config.get("alpha_state", 0.3)
            alpha_tool = jig_config.get("alpha_tool", 0.4)
            alpha_transfer = jig_config.get("alpha_transfer", 0.2)

            self.jig_computer = JointInformationGain(
                alpha_state=alpha_state,
                alpha_tool=alpha_tool,
                alpha_transfer=alpha_transfer,
            )

            exploration_window = jig_config.get("exploration_window", 50)
            trend_threshold = jig_config.get("trend_threshold", -0.01)
            self.sustained_bonus = SustainedExplorationBonus(
                window=exploration_window,
                trend_threshold=trend_threshold,
            )

            # 课程学习调度器
            total_steps = jig_config.get("total_steps", 300)
            beta = jig_config.get("curriculum_beta", 1.0)
            self.curriculum_scheduler = CurriculumScheduler(
                alpha_0=1.0,
                total_steps=total_steps,
                beta=beta,
            )

            logger.info(
                f"[TauBenchInteractionIG] JIG mode enabled: "
                f"alpha_state={alpha_state}, alpha_tool={alpha_tool}, "
                f"alpha_transfer={alpha_transfer}, total_steps={total_steps}"
            )
        else:
            if self.reward_mode in _REWARD_FUNCTIONS:
                self._compute_reward = _REWARD_FUNCTIONS[self.reward_mode]
            else:
                logger.warning(f"Unknown reward_mode '{self.reward_mode}', using binary")
                self._compute_reward = _compute_binary_reward
            self.jig_computer = None
            self.sustained_bonus = None
            self.curriculum_scheduler = None

        logger.info(f"[TauBenchInteractionIG] env_name={self.env_name}, reward_mode={self.reward_mode}, use_jig={self.use_jig}")

        self._instance_dict: dict[str, dict] = {}

    async def start_interaction(
        self,
        instance_id: Optional[str] = None,
        task_id: int = 0,
        **kwargs,
    ) -> str:
        """
        ToolAgentLoop 里每条 trajectory 开始时调用一次。
        在这里创建 env 实例并绑定到 contextvar。
        """
        if instance_id is None:
            instance_id = str(uuid.uuid4())

        # 延迟 import,避免单测时强依赖 tau_bench 包
        from tau_bench.envs import get_env

        task_id_int = int(task_id)
        env = get_env(
            env_name=self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_api_base=self.user_base_url,
            task_split=self.task_split,
            task_index=task_id_int,
        )
        env.reset(task_index=task_id_int)

        state = make_initial_state(task_id_int)

        # IG-GRPO: 添加课程学习权重
        if self.use_jig and self.curriculum_scheduler:
            state["curriculum_alpha"] = self.curriculum_scheduler.get_alpha()

        # 绑定到当前 asyncio task 的 context
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)

        self._instance_dict[instance_id] = {"env": env, "state": state}

        logger.debug(
            f"[start_interaction] instance={instance_id[:8]} task_id={task_id_int}"
        )
        return instance_id

    async def generate_response(
        self,
        instance_id: str,
        messages: list[dict[str, Any]],
        **kwargs,
    ) -> tuple[bool, str, float, dict[str, Any]]:
        """
        被 ToolAgentLoop 在 AgentState.INTERACTING 触发。

        Returns:
            (should_terminate, user_response_content, reward, metadata)
        """
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            raise RuntimeError(
                f"[CRITICAL] TauBenchInteractionIG.generate_response called for "
                f"instance_id={instance_id} but no corresponding entry found."
            )

        env = entry["env"]
        state = entry["state"]

        # Defensive re-set
        CURRENT_TAU_ENV.set(env)
        CURRENT_TAU_STATE.set(state)

        assistant_content = _extract_latest_assistant_content(messages)

        # 记录 implicit think
        if assistant_content and len(assistant_content) > 100:
            last_action = state["action_history"][-1] if state["action_history"] else None
            is_duplicate = (
                last_action
                and last_action.get("tool") == "implicit_think"
                and last_action.get("content", "") == assistant_content[:300]
            )
            if not is_duplicate:
                state["action_history"].append({
                    "tool": "implicit_think",
                    "parameters": {},
                    "param_str": "",
                    "inc_reward": 0,
                    "done": False,
                    "is_error": False,
                    "content": assistant_content[:300],
                })

        # 污染检测
        if _has_forbidden_token(assistant_content):
            state["contaminated"] = True
            state["done"] = True
            logger.info(
                f"[generate_response] FORBIDDEN_TOKEN detected in task {state['task_id']}"
            )
            return (
                True,
                "",
                0.0,
                {
                    "contaminated": True,
                    "reason": "forbidden_template_token",
                    "total_reward": state["total_reward"],
                    "num_turns": state.get("num_turns", 0) + state.get("num_tool_calls", 0),
                    "task_id": state["task_id"],
                },
            )

        # 正常路径: 驱动 user simulator
        from tau_bench.types import Action, RESPOND_ACTION_NAME

        try:
            action = Action(
                name=RESPOND_ACTION_NAME,
                kwargs={"content": assistant_content},
            )
            step_res = env.step(action)
        except Exception as e:
            logger.warning(
                f"[generate_response] env.step(RESPOND) failed: {type(e).__name__}: {e}"
            )
            state["done"] = True
            return (
                True,
                "",
                0.0,
                {
                    "error": "respond_exception",
                    "reason": f"{type(e).__name__}: {e}",
                    "task_id": state["task_id"],
                },
            )

        inc_reward = float(getattr(step_res, "reward", 0.0))
        is_done = bool(getattr(step_res, "done", False))
        state["total_reward"] += inc_reward
        state["num_turns"] = state.get("num_turns", 0) + 1

        total_turns = state.get("num_turns", 0) + state.get("num_tool_calls", 0)

        # 终止条件: env 说 done / 超 max_turns
        if is_done or total_turns >= self.max_turns:
            state["done"] = True
            final_score = self._compute_final_reward(state)
            return (
                True,
                "",
                final_score,
                {
                    "total_reward": state["total_reward"],
                    "num_turns": total_turns,
                    "num_tool_calls": state.get("num_tool_calls", 0),
                    "task_id": state["task_id"],
                    "reason": "done" if is_done else "max_turns",
                    "reward_mode": self.reward_mode,
                    "jig_stats": self._get_jig_stats(),
                },
            )

        # 继续交互: 返回 user reply
        user_reply = str(getattr(step_res, "observation", ""))
        return (
            False,
            user_reply,
            0.0,
            {
                "turn": total_turns,
                "task_id": state["task_id"],
            },
        )

    def _compute_final_reward(self, state: dict) -> float:
        """计算最终奖励（含 JIG）"""
        # 基础奖励
        base_reward = 1.0 if state["total_reward"] >= 1.0 else 0.0

        if not self.use_jig or self.jig_computer is None:
            if hasattr(self, '_compute_reward'):
                return self._compute_reward(state)
            return base_reward

        # 计算 JIG 奖励
        return _compute_jig_reward(state, self.jig_computer, self.sustained_bonus)

    def _get_jig_stats(self) -> dict:
        """获取 JIG 统计"""
        if not self.use_jig or self.jig_computer is None:
            return {}

        coverage_stats = self.jig_computer.coverage_tracker.get_coverage_stats()
        sustained_stats = self.sustained_bonus.get_stats() if self.sustained_bonus else {}

        return {
            "coverage": coverage_stats,
            "sustained": sustained_stats,
            "curriculum_alpha": self.curriculum_scheduler.get_alpha() if self.curriculum_scheduler else 0.0,
        }

    async def calculate_score(self, instance_id: str, **kwargs) -> float | dict:
        """Turn-level score"""
        entry = self._instance_dict.get(instance_id)
        if entry is None:
            return {"score": 0.0, "outcome_score": 0.0}

        state = entry["state"]
        outcome = 1.0 if state["total_reward"] >= 1.0 else 0.0

        if self.use_jig:
            jig_score = self._compute_final_reward(state)
            return {
                "score": jig_score,
                "outcome_score": outcome,
                "jig_score": jig_score - outcome,
            }

        return {"score": outcome, "outcome_score": outcome}

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        """Trajectory 结束时清理"""
        entry = self._instance_dict.pop(instance_id, None)
        if entry and self.use_jig and self.jig_computer:
            # 重置 episode 级状态（保留长期记忆）
            self.jig_computer.coverage_tracker.reset_episode()

    def step_curriculum(self) -> None:
        """步进课程学习调度器"""
        if self.use_jig and self.curriculum_scheduler:
            self.curriculum_scheduler.step()

    def get_jig_stats(self) -> dict:
        """获取 JIG 统计"""
        return self._get_jig_stats()
