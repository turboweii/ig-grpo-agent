"""
Entropy@K 评测框架 - IG-GRPO 核心评测
多维度探索质量评测
"""
from __future__ import annotations
import json
import math
from collections import Counter
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EntropyAtKResult:
    """Entropy@K 评测结果"""
    task_id: int
    success_rate: float           # 任务成功率
    state_coverage: float         # 状态覆盖率
    tool_entropy: float           # 工具多样性熵
    path_diversity: float         # 路径多样性
    avg_path_length: float        # 平均路径长度
    avg_tool_calls: float         # 平均工具调用数
    unique_state_tools: int       # 唯一 state-tool 对数量
    redundancy_rate: float        # 冗余调用率


@dataclass
class AggregatedEntropyResult:
    """聚合评测结果"""
    overall_success_rate: float
    overall_state_coverage: float
    overall_tool_entropy: float
    overall_path_diversity: float
    per_task_results: List[EntropyAtKResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overall_success_rate": self.overall_success_rate,
            "overall_state_coverage": self.overall_state_coverage,
            "overall_tool_entropy": self.overall_tool_entropy,
            "overall_path_diversity": self.overall_path_diversity,
            "per_task_results": [
                {
                    "task_id": r.task_id,
                    "success_rate": r.success_rate,
                    "state_coverage": r.state_coverage,
                    "tool_entropy": r.tool_entropy,
                    "path_diversity": r.path_diversity,
                    "avg_path_length": r.avg_path_length,
                    "avg_tool_calls": r.avg_tool_calls,
                    "unique_state_tools": r.unique_state_tools,
                    "redundancy_rate": r.redundancy_rate,
                }
                for r in self.per_task_results
            ],
        }


class EntropyAtKEvaluator:
    """
    Entropy@K 评测器
    在 K 次采样中评测探索质量
    """

    def __init__(
        self,
        state_space_size: int = 1_000_000,  # 估计的状态空间大小
        k: int = 4,
    ):
        self.state_space_size = state_space_size
        self.k = k

    def _levenshtein_distance(self, path1: List[str], path2: List[str]) -> int:
        """计算两条路径的编辑距离"""
        m, n = len(path1), len(path2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if path1[i - 1] == path2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i - 1][j],
                        dp[i][j - 1],
                        dp[i - 1][j - 1],
                    )

        return dp[m][n]

    def _compute_path_diversity(self, trajectories: List[List[str]]) -> float:
        """
        计算路径多样性
        使用平均编辑距离的归一化值
        """
        if len(trajectories) < 2:
            return 0.0

        # 提取工具序列
        tool_sequences = []
        for traj in trajectories:
            tools = []
            for step in traj:
                if hasattr(step, "tool_name") and step.tool_name:
                    tools.append(step.tool_name)
            tool_sequences.append(tools)

        # 计算所有对的编辑距离
        distances = []
        for i in range(len(tool_sequences)):
            for j in range(i + 1, len(tool_sequences)):
                dist = self._levenshtein_distance(tool_sequences[i], tool_sequences[j])
                max_len = max(len(tool_sequences[i]), len(tool_sequences[j]))
                if max_len > 0:
                    normalized_dist = dist / max_len
                    distances.append(normalized_dist)

        return np.mean(distances) if distances else 0.0

    def _compute_tool_entropy(self, tool_counts: Counter) -> float:
        """计算工具使用分布的熵"""
        total = sum(tool_counts.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in tool_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)

        return entropy

    def _compute_redundancy_rate(self, trajectories: List) -> float:
        """
        计算冗余调用率
        定义：3 轮内重复调用同一工具的比例
        """
        redundant_calls = 0
        total_calls = 0

        for traj in trajectories:
            recent_calls = []
            for step in traj:
                if hasattr(step, "tool_name") and step.tool_name:
                    total_calls += 1
                    if step.tool_name in recent_calls[-3:]:
                        redundant_calls += 1
                    recent_calls.append(step.tool_name)

        return redundant_calls / total_calls if total_calls > 0 else 0.0

    def evaluate_task(
        self,
        task_id: int,
        trajectories: List,  # K 条轨迹
    ) -> EntropyAtKResult:
        """
        评测单个任务的 Entropy@K
        """
        # 统计
        successes = sum(1 for t in trajectories if t.success)
        success_rate = successes / len(trajectories)

        # 状态覆盖
        visited_states = set()
        for traj in trajectories:
            if hasattr(traj, "visited_states"):
                visited_states.update(traj.visited_states)
            else:
                # 从 steps 提取
                for step in traj.steps:
                    if hasattr(step, "state_hash") and step.state_hash:
                        visited_states.add(step.state_hash)

        state_coverage = len(visited_states) / self.state_space_size

        # 工具统计
        tool_counts = Counter()
        total_tool_calls = 0
        total_path_length = 0
        state_tool_pairs = set()

        for traj in trajectories:
            if hasattr(traj, "state_tool_pairs"):
                state_tool_pairs.update(traj.state_tool_pairs)

            for step in traj.steps:
                if hasattr(step, "tool_name") and step.tool_name:
                    tool_counts[step.tool_name] += 1
                    total_tool_calls += 1

            total_path_length += len(traj.steps)

        tool_entropy = self._compute_tool_entropy(tool_counts)
        path_diversity = self._compute_path_diversity(trajectories)
        redundancy_rate = self._compute_redundancy_rate(trajectories)

        return EntropyAtKResult(
            task_id=task_id,
            success_rate=success_rate,
            state_coverage=state_coverage,
            tool_entropy=tool_entropy,
            path_diversity=path_diversity,
            avg_path_length=total_path_length / len(trajectories),
            avg_tool_calls=total_tool_calls / len(trajectories),
            unique_state_tools=len(state_tool_pairs),
            redundancy_rate=redundancy_rate,
        )

    def evaluate(
        self,
        results: Dict[int, List],  # task_id -> K 条轨迹
    ) -> AggregatedEntropyResult:
        """
        评测所有任务
        """
        per_task_results = []

        for task_id, trajectories in results.items():
            result = self.evaluate_task(task_id, trajectories)
            per_task_results.append(result)

        # 聚合
        overall_success_rate = np.mean([r.success_rate for r in per_task_results])
        overall_state_coverage = np.mean([r.state_coverage for r in per_task_results])
        overall_tool_entropy = np.mean([r.tool_entropy for r in per_task_results])
        overall_path_diversity = np.mean([r.path_diversity for r in per_task_results])

        return AggregatedEntropyResult(
            overall_success_rate=overall_success_rate,
            overall_state_coverage=overall_state_coverage,
            overall_tool_entropy=overall_tool_entropy,
            overall_path_diversity=overall_path_diversity,
            per_task_results=per_task_results,
        )


class SemanticPartitioner:
    """
    语义任务分割器
    用于 Anti-Leakage 评测：基于任务语义分割 covered/uncovered/unseen
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(model_name)
        except ImportError:
            self.encoder = None
            print("Warning: sentence_transformers not available, using hash-based partitioning")

    def partition(
        self,
        tasks: List[Dict],
        n_clusters: int = 5,
        train_ratio: float = 0.8,
    ) -> Dict[str, List]:
        """
        将任务分割为 covered_seen / uncovered_seen / unseen
        """
        if self.encoder is None:
            return self._hash_based_partition(tasks, train_ratio)

        # 获取任务描述
        descriptions = [t.get("description", "") for t in tasks]
        embeddings = self.encoder.encode(descriptions)

        # 聚类
        from sklearn.cluster import KMeans
        clusters = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(embeddings)

        # 分割
        partitioned = {"covered_seen": [], "uncovered_seen": [], "unseen": []}

        for cluster_id in range(n_clusters):
            cluster_tasks = [t for t, c in zip(tasks, clusters) if c == cluster_id]
            import random
            random.shuffle(cluster_tasks)

            split = int(len(cluster_tasks) * train_ratio)
            train_tasks = cluster_tasks[:split]
            test_tasks = cluster_tasks[split:]

            # 检查是否有教师轨迹
            for task in train_tasks:
                if task.get("has_teacher", False):
                    partitioned["covered_seen"].append(task)
                else:
                    partitioned["uncovered_seen"].append(task)

            partitioned["unseen"].extend(test_tasks)

        return partitioned

    def _hash_based_partition(self, tasks: List, train_ratio: float) -> Dict[str, List]:
        """基于哈希的简单分割（fallback）"""
        import hashlib
        import random

        partitioned = {"covered_seen": [], "uncovered_seen": [], "unseen": []}

        for task in tasks:
            hash_val = int(hashlib.md5(task.get("description", "").encode()).hexdigest(), 16)
            if hash_val % 10 < train_ratio * 10:
                if task.get("has_teacher", False):
                    partitioned["covered_seen"].append(task)
                else:
                    partitioned["uncovered_seen"].append(task)
            else:
                partitioned["unseen"].append(task)

        return partitioned


def evaluate_checkpoint(
    checkpoint_path: str,
    tasks: List,
    evaluator: EntropyAtKEvaluator,
    k: int = 4,
) -> AggregatedEntropyResult:
    """
    评测单个检查点
    """
    # 加载模型
    # ... (根据具体框架实现)

    # 运行 K 次采样
    results = {}
    for task in tasks:
        task_trajectories = []
        for _ in range(k):
            # 运行单条轨迹
            # trajectory = run_single_task(task, policy)
            # task_trajectories.append(trajectory)
            pass
        results[task["id"]] = task_trajectories

    return evaluator.evaluate(results)
