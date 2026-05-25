"""
JIG (Joint Information Gain) Components for IG-GRPO
This module contains the core components for Information Gain guided exploration.

Key components:
1. HierarchicalCoverageTracker - Three-layer Bloom Filter for state tracking
2. JointInformationGain - Joint Information Gain calculator
3. SustainedExplorationBonus - Exploration sustainability monitor
"""
from __future__ import annotations

import math
import hashlib
import json
from typing import Set, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass

try:
    from pybloom_live import BloomFilter, ScalableBloomFilter
    BLOOM_AVAILABLE = True
except ImportError:
    # Fallback to set-based implementation if pybloom_live not available
    BLOOM_AVAILABLE = False
    BloomFilter = None
    ScalableBloomFilter = None


@dataclass
class CoverageReward:
    """Coverage reward breakdown"""
    state_new: float = 0.0      # New state reward
    tool_new: float = 0.0       # New tool reward (in that state)
    combo_new: float = 0.0      # New state-tool combination reward
    total: float = 0.0          # Total reward


class HierarchicalCoverageTracker:
    """
    Three-layer coverage tracker for state-space exploration.
    - Short-term: Current episode states (with decay)
    - Mid-term: Recent N episode states
    - Long-term: Entire training history states

    Falls back to set-based implementation if pybloom_live not available.
    """

    def __init__(
        self,
        short_capacity: int = 100_000,
        mid_capacity: int = 1_000_000,
        long_capacity: int = 10_000_000,
        short_error: float = 0.001,
        mid_error: float = 0.01,
        long_error: float = 0.05,
        decay_interval: int = 100,
    ):
        self._short_capacity = short_capacity
        self._mid_capacity = mid_capacity
        self._long_capacity = long_capacity
        self._short_error = short_error
        self._mid_error = mid_error
        self._long_error = long_error

        if BLOOM_AVAILABLE:
            self.short_term = BloomFilter(capacity=short_capacity, error_rate=short_error)
            self.mid_term = BloomFilter(capacity=mid_capacity, error_rate=mid_error)
            self.long_term = ScalableBloomFilter(initial_capacity=long_capacity, error_rate=long_error)
            self._use_bloom = True
        else:
            # Fallback to set-based implementation
            self.short_term = set()
            self.mid_term = set()
            self.long_term = set()
            self._use_bloom = False

        # State-tool mapping (for exact queries)
        self.state_tools: Dict[int, Set[str]] = defaultdict(set)

        # Statistics
        self.decay_counter = 0
        self.decay_interval = decay_interval

        # Reward weights (tunable via curriculum learning)
        self.state_reward_weight = 0.3
        self.tool_reward_weight = 0.4
        self.combo_reward_weight = 0.3

    def _hash_state(self, state: Any) -> int:
        """State hashing function"""
        if isinstance(state, int):
            return state
        if isinstance(state, str):
            return int(hashlib.md5(state.encode()).hexdigest()[:8], 16)
        if isinstance(state, dict):
            return int(hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()[:8], 16)
        return hash(state) % (10 ** 8)

    def add(self, state: Any, tool: Optional[str] = None) -> CoverageReward:
        """
        Add state (optional tool) to tracker and return coverage reward breakdown.
        """
        state_hash = self._hash_state(state)

        reward = CoverageReward()

        # Check new state
        is_new_state = state_hash not in self.long_term
        if is_new_state:
            reward.state_new = self.state_reward_weight

        # Check new tool combination
        if tool:
            is_new_tool = tool not in self.state_tools[state_hash]
            if is_new_tool:
                reward.tool_new = self.tool_reward_weight

            combo_key = f"{state_hash}:{tool}"
            is_new_combo = combo_key not in self.long_term
            if is_new_combo:
                reward.combo_new = self.combo_reward_weight

            self.state_tools[state_hash].add(tool)

        # Add to all layers
        self.short_term.add(state_hash)
        if tool:
            self.long_term.add(f"{state_hash}:{tool}")
        else:
            self.long_term.add(state_hash)

        # Periodic decay
        self.decay_counter += 1
        if self.decay_counter >= self.decay_interval:
            self._decay()

        reward.total = reward.state_new + reward.tool_new + reward.combo_new
        return reward

    def _decay(self):
        """Periodic decay: short-term -> mid-term -> long-term -> discard"""
        if self._use_bloom:
            # Bloom Filter doesn't support deletion, so we recreate
            self.short_term = BloomFilter(
                capacity=self._short_capacity,
                error_rate=self._short_error
            )
        else:
            # Set-based implementation supports clear
            self.short_term.clear()
        self.decay_counter = 0

    def contains(self, state: Any, tool: Optional[str] = None) -> bool:
        """Check if state/combo has been visited"""
        state_hash = self._hash_state(state)
        if tool:
            return f"{state_hash}:{tool}" in self.long_term
        return state_hash in self.long_term

    def get_state_tools(self, state: Any) -> Set[str]:
        """Get set of tools already used for given state"""
        state_hash = self._hash_state(state)
        return self.state_tools.get(state_hash, set()).copy()

    def get_coverage_stats(self) -> dict:
        """Get coverage statistics"""
        return {
            "unique_states": len(self.state_tools),
            "total_state_tool_pairs": sum(len(tools) for tools in self.state_tools.values()),
            "decay_counter": self.decay_counter,
            "use_bloom": self._use_bloom,
        }

    def reset_episode(self):
        """Reset episode-level state (doesn't affect long-term memory)"""
        if self._use_bloom:
            self.short_term = BloomFilter(
                capacity=self._short_capacity,
                error_rate=self._short_error
            )
        else:
            self.short_term.clear()

    def reset_all(self):
        """Complete reset"""
        if self._use_bloom:
            self.short_term = BloomFilter(
                capacity=self._short_capacity,
                error_rate=self._short_error
            )
            self.mid_term = BloomFilter(
                capacity=self._mid_capacity,
                error_rate=self._mid_error
            )
            self.long_term = ScalableBloomFilter(
                initial_capacity=self._long_capacity,
                error_rate=self._long_error
            )
        else:
            self.short_term.clear()
            self.mid_term.clear()
            self.long_term.clear()
        self.state_tools.clear()
        self.decay_counter = 0


class JointInformationGain:
    """
    Joint Information Gain Calculator for IG-GRPO

    JIG = H(State, Tool) - H(State, Tool | state_t, tool_t)

    This computes the information gain from taking a specific action
    in a specific state, encouraging exploration of novel state-action pairs.
    """

    def __init__(
        self,
        alpha_state: float = 0.3,
        alpha_tool: float = 0.4,
        alpha_transfer: float = 0.2,
        alpha_sustained: float = 0.1,
    ):
        self.alpha_state = alpha_state
        self.alpha_tool = alpha_tool
        self.alpha_transfer = alpha_transfer
        self.alpha_sustained = alpha_sustained

        self.coverage_tracker = HierarchicalCoverageTracker()

        # Tool transition tracking
        self.tool_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        self.total_transitions = 0

    def compute_jig(
        self,
        state: Any,
        tool: str,
        prev_tool: Optional[str] = None,
    ) -> float:
        """
        Compute Joint Information Gain reward.

        Args:
            state: Current state representation
            tool: Current tool being used
            prev_tool: Previous tool used (for transition novelty)

        Returns:
            JIG reward value
        """
        # Component 1: State novelty (via coverage tracker)
        coverage_reward = self.coverage_tracker.add(state, tool)
        state_novelty = coverage_reward.total

        # Component 2: Tool transition novelty
        transfer_novelty = 0.0
        if prev_tool:
            key = (prev_tool, tool)
            self.tool_transitions[key] += 1
            self.total_transitions += 1

            # Rarer transition = higher reward
            count = self.tool_transitions[key]
            transfer_novelty = 1.0 / math.sqrt(count) if count > 0 else 1.0

        # Component 3: Combined weighting
        jig = (
            self.alpha_state * state_novelty +
            self.alpha_tool * coverage_reward.tool_new +
            self.alpha_transfer * transfer_novelty
        )

        # Component 4: Sustained exploration (handled externally)
        # sustained_bonus provided by SustainedExplorationBonus

        return jig

    def get_transfer_probability(self, tool_i: str, tool_j: str) -> float:
        """Get tool transition probability"""
        if self.total_transitions == 0:
            return 0.0
        count = self.tool_transitions.get((tool_i, tool_j), 0)
        return count / self.total_transitions

    def reset(self):
        """Reset all state"""
        self.coverage_tracker.reset_all()
        self.tool_transitions.clear()
        self.total_transitions = 0


class SustainedExplorationBonus:
    """
    Sustained Exploration Bonus for IG-GRPO

    Monitors IG trend and provides additional exploration incentive
    when IG starts to plateau or decline, preventing premature convergence.
    """

    def __init__(self, window: int = 50, trend_threshold: float = -0.01):
        self.window = window
        self.trend_threshold = trend_threshold
        self.ig_history: list = []
        self.episode_count = 0

    def update(self, ig_value: float):
        """Update IG history with new value"""
        self.ig_history.append(ig_value)
        if len(self.ig_history) > self.window:
            self.ig_history.pop(0)
        self.episode_count += 1

    def compute_bonus(self) -> float:
        """
        Compute sustained exploration bonus.

        Returns bonus if IG is consistently declining.
        """
        if len(self.ig_history) < self.window // 2:
            return 0.0

        # Compute linear trend (simple version)
        recent_avg = sum(self.ig_history[-self.window//2:]) / (self.window // 2)
        earlier_avg = sum(self.ig_history[:self.window//2]) / (self.window // 2)
        trend = (recent_avg - earlier_avg) / (self.window // 2)

        # IG consistently declining -> give bonus
        if trend < self.trend_threshold:
            return 0.5 * abs(trend)  # Bonus proportional to decline magnitude

        return 0.0

    def force_diverse_action(
        self,
        state: Any,
        available_tools: list[str],
        state_tool_visits: Dict[Tuple[Any, str], int],
    ) -> Optional[str]:
        """
        Force selection of less commonly used tool (similar to ε-greedy).

        Returns a tool to use or None if no forced action needed.
        """
        import random

        if random.random() < 0.1:  # 10% probability
            # Find least used tool for this state
            if not available_tools:
                return None

            tool_counts = []
            for tool in available_tools:
                key = (state, tool)
                count = state_tool_visits.get(key, 0)
                tool_counts.append((tool, count))

            # Select least used
            min_count = min(count for _, count in tool_counts)
            candidates = [tool for tool, count in tool_counts if count == min_count]

            return random.choice(candidates) if candidates else None

        return None

    def get_stats(self) -> dict:
        """Get statistics"""
        return {
            "episode_count": self.episode_count,
            "current_window_size": len(self.ig_history),
            "avg_ig": sum(self.ig_history) / len(self.ig_history) if self.ig_history else 0.0,
        }


class CurriculumScheduler:
    """
    Curriculum Learning Scheduler for IG-GRPO

    Adjusts exploration weight over training time:
    α(t) = α_0 × (1 - t/T)^β

    Early training: high exploration weight
    Late training: low exploration weight (focus on exploitation)
    """

    def __init__(
        self,
        alpha_0: float = 1.0,
        total_steps: int = 1000,
        beta: float = 1.0,
    ):
        self.alpha_0 = alpha_0
        self.total_steps = total_steps
        self.beta = beta
        self.current_step = 0

    def get_alpha(self, step: Optional[int] = None) -> float:
        """Get current alpha value for exploration weight"""
        if step is None:
            step = self.current_step
        else:
            self.current_step = step

        if step >= self.total_steps:
            return 0.0

        progress = step / self.total_steps
        return self.alpha_0 * ((1 - progress) ** self.beta)

    def step(self):
        """Increment step counter"""
        self.current_step += 1

    def reset(self):
        """Reset step counter"""
        self.current_step = 0
