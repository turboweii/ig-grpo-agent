"""
τ-bench 环境封装
统一 airline/retail 两个子集的接口,为评测和 IG-GRPO rollout 提供统一 API
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Any
import json
import os
from collections import Counter

# τ-bench 原生 API
from tau_bench.envs import get_env
from tau_bench.types import EnvRunResult, Action, RESPOND_ACTION_NAME


@dataclass
class TrajectoryStep:
    """一轮交互的完整记录"""
    turn_idx: int
    role: str  # "user" | "assistant" | "tool"
    content: str
    tool_calls: Optional[list[dict]] = None
    tool_name: Optional[str] = None
    state_hash: Optional[int] = None  # IG-GRPO: 用于追踪状态覆盖


@dataclass
class TrajectoryResult:
    """一条完整轨迹的结果,用于评测和 RL 训练"""
    task_id: int
    success: bool           # outcome reward (0/1)
    reward: float           # τ-bench 原生 reward
    num_turns: int
    num_tool_calls: int
    steps: list[TrajectoryStep] = field(default_factory=list)
    raw_messages: list[dict] = field(default_factory=list)  # OpenAI 格式
    error: Optional[str] = None
    # IG-GRPO: 覆盖率追踪
    state_tool_pairs: set[tuple[int, str]] = field(default_factory=set)
    visited_states: set[int] = field(default_factory=set)
    tool_diversity_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "reward": self.reward,
            "num_turns": self.num_turns,
            "num_tool_calls": self.num_tool_calls,
            "raw_messages": self.raw_messages,
            "error": self.error,
            "state_tool_pairs": list(self.state_tool_pairs),
            "visited_states": list(self.visited_states),
            "tool_diversity_score": self.tool_diversity_score,
        }


class TauBenchWrapper:
    """
    对 τ-bench 的薄封装,提供两个关键能力:
    1. run_single_task: 给定 policy 和 task_id,跑一条轨迹
    2. batch_eval: 批量评测,用于 baseline 测试

    policy 需要实现 __call__(messages: list[dict]) -> dict 接口,
    返回 OpenAI 格式的 assistant message (可能包含 tool_calls).
    """

    def __init__(
        self,
        env_name: str = "retail",         # IG-GRPO: 默认用 retail (40 工具)
        user_strategy: str = "llm",
        user_model: str = "qwen2.5-72b-awq",
        user_provider: str = "local",
        user_base_url: Optional[str] = "http://localhost:8001/v1",
        task_split: str = "train",
        task_index: Optional[int] = None,
    ):
        self.env_name = env_name
        self.user_strategy = user_strategy
        self.user_model = user_model
        self.user_provider = user_provider
        self.user_base_url = user_base_url
        self.task_split = task_split
        self.task_index = task_index

    def _make_env(self, task_idx: int):
        """为每个 task 创建一个独立的 env 实例"""
        return get_env(
            env_name=self.env_name,
            user_strategy=self.user_strategy,
            user_model=self.user_model,
            user_provider=self.user_provider,
            user_api_base=self.user_base_url,
            task_split=self.task_split,
            task_index=task_idx,
        )

    def get_num_tasks(self) -> int:
        env = self._make_env(0)
        return len(env.tasks)

    def _hash_state(self, messages: list[dict], tool_name: Optional[str] = None) -> int:
        """
        IG-GRPO: 状态哈希函数
        结合对话历史和最近工具调用结果生成状态指纹
        """
        # 取最近 3 轮对话 + 当前工具名
        recent_msgs = messages[-6:] if len(messages) >= 6 else messages
        state_str = json.dumps(recent_msgs, sort_keys=True)
        if tool_name:
            state_str += f"|{tool_name}"
        return hash(state_str) % (10 ** 8)  # 限制在合理范围

    def run_single_task(
        self,
        task_idx: int,
        policy,
        max_turns: int = 30,
        track_coverage: bool = True,  # IG-GRPO: 是否追踪覆盖率
    ) -> TrajectoryResult:
        env = self._make_env(task_idx)
        if hasattr(policy, "set_tools"):
            policy.set_tools(env.tools_info)
        obs_res = env.reset(task_index=task_idx)

        # IG-GRPO: 初始化覆盖率追踪
        state_tool_pairs: set[tuple[int, str]] = set()
        visited_states: set[int] = set()
        tool_counts: Counter = Counter()

        # Date grounding (tau-bench 要求 2024 年)
        if self.env_name == "retail":
            system_content = (
                "# Current Date Context\n"
                "The current date is 2024-05-15 (Wednesday). "
                "When users mention dates without specifying the year, "
                "always assume they refer to 2024. "
                "All product orders and exchanges should use 2024 dates unless explicitly stated otherwise."
            )
        else:
            system_content = (
                "# Current Date Context\n"
                "The current date is 2024-05-15 (Wednesday). "
                "When users mention dates without specifying the year, "
                "always assume they refer to 2024. "
                "All flight searches and reservations should use 2024 dates unless explicitly stated otherwise."
            )

        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": str(obs_res.observation)},
        ]
        steps: list[TrajectoryStep] = []
        total_reward = 0.0
        done = False
        error_msg = None
        turn_idx = 0
        tool_call_count = 0
        recent_tool_calls: list[tuple[str, str]] = []
        prev_tool = None

        try:
            while not done and turn_idx < max_turns:
                assistant_msg = policy(messages)
                messages.append(assistant_msg)

                # IG-GRPO: 计算当前状态哈希
                state_hash = self._hash_state(messages) if track_coverage else None

                steps.append(TrajectoryStep(
                    turn_idx=turn_idx,
                    role="assistant",
                    content=assistant_msg.get("content", "") or "",
                    tool_calls=assistant_msg.get("tool_calls"),
                    state_hash=state_hash,
                ))

                tcs = assistant_msg.get("tool_calls")
                if tcs:
                    # Loop detection
                    for tc in tcs:
                        call_sig = (tc["function"]["name"], tc["function"]["arguments"])
                        recent_tool_calls.append(call_sig)
                    if len(recent_tool_calls) > 20:
                        recent_tool_calls = recent_tool_calls[-20:]
                    call_counts = Counter(recent_tool_calls)
                    if any(c >= 3 for c in call_counts.values()):
                        error_msg = "Loop detected: same tool call repeated 3+ times"
                        break

                    for tc in tcs:
                        tool_name = tc["function"]["name"]
                        tool_call_count += 1
                        tool_counts[tool_name] += 1

                        # IG-GRPO: 追踪 state-tool 组合
                        if track_coverage and state_hash is not None:
                            state_tool_pairs.add((state_hash, tool_name))
                            visited_states.add(state_hash)

                        action = Action(
                            name=tool_name,
                            kwargs=json.loads(tc["function"]["arguments"]),
                        )
                        tool_result = env.step(action)
                        obs_content = tool_result.observation if hasattr(tool_result, 'observation') else str(tool_result)

                        tool_msg = {
                            "role": "tool",
                            "tool_call_id": tc.get("id", f"call_{tool_call_count}"),
                            "name": tool_name,
                            "content": str(obs_content),
                        }
                        messages.append(tool_msg)
                        steps.append(TrajectoryStep(
                            turn_idx=turn_idx, role="tool",
                            content=tool_msg["content"], tool_name=tool_name,
                            state_hash=self._hash_state(messages) if track_coverage else None,
                        ))
                        total_reward += getattr(tool_result, 'reward', 0.0)
                        if getattr(tool_result, 'done', False):
                            done = True
                            break
                    prev_tool = tool_name
                else:
                    action = Action(
                        name=RESPOND_ACTION_NAME,
                        kwargs={"content": assistant_msg.get("content", "")},
                    )
                    user_obs = env.step(action)
                    if getattr(user_obs, 'done', False):
                        done = True
                        total_reward += getattr(user_obs, 'reward', 0.0)
                    else:
                        obs_str = getattr(user_obs, 'observation', str(user_obs))
                        user_msg = {"role": "user", "content": obs_str}
                        messages.append(user_msg)
                        steps.append(TrajectoryStep(
                            turn_idx=turn_idx, role="user", content=obs_str,
                        ))
                    prev_tool = None
                turn_idx += 1
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        # IG-GRPO: 计算工具多样性熵
        tool_diversity = 0.0
        if track_coverage and tool_counts:
            total = sum(tool_counts.values())
            for count in tool_counts.values():
                p = count / total
                tool_diversity -= p * (0 if p == 0 else p * __import__('math').log(p))

        success = total_reward >= 1.0
        return TrajectoryResult(
            task_id=task_idx, success=success, reward=total_reward,
            num_turns=turn_idx, num_tool_calls=tool_call_count,
            steps=steps, raw_messages=messages, error=error_msg,
            state_tool_pairs=state_tool_pairs,
            visited_states=visited_states,
            tool_diversity_score=tool_diversity,
        )
