"""
Pass@K 评测框架 - IG-GRPO
标准的 pass@k 评测，区分 covered/uncovered/unseen 任务
"""
from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class PassKResult:
    """单个任务的评测结果"""
    task_id: int
    task_type: str  # "covered_seen" | "uncovered_seen" | "unseen"
    success: bool
    num_samples: int
    num_turns: int
    num_tool_calls: int
    error: Optional[str] = None


@dataclass
class AggregatedPassKResult:
    """聚合 Pass@K 结果"""
    overall_pass_at_1: float
    covered_seen_pass_at_1: float
    uncovered_seen_pass_at_1: float
    unseen_pass_at_1: float
    generalization_pass_at_1: float  # (uncovered_seen + unseen) 泛化指标
    total_tasks: int
    per_task_results: List[PassKResult]

    def to_dict(self) -> dict:
        return {
            "overall_pass_at_1": self.overall_pass_at_1,
            "covered_seen_pass_at_1": self.covered_seen_pass_at_1,
            "uncovered_seen_pass_at_1": self.uncovered_seen_pass_at_1,
            "unseen_pass_at_1": self.unseen_pass_at_1,
            "generalization_pass_at_1": self.generalization_pass_at_1,
            "total_tasks": self.total_tasks,
            "per_task_results": [
                {
                    "task_id": r.task_id,
                    "task_type": r.task_type,
                    "success": r.success,
                    "num_samples": r.num_samples,
                    "num_turns": r.num_turns,
                    "num_tool_calls": r.num_tool_calls,
                    "error": r.error,
                }
                for r in self.per_task_results
            ],
        }


class PassKEvaluator:
    """
    Pass@K 评测器
    标准的任务成功率评测
    """

    def __init__(
        self,
        task_partition: Optional[Dict[str, List]] = None,
    ):
        """
        Args:
            task_partition: 任务分割，包含 covered_seen/uncovered_seen/unseen
        """
        self.task_partition = task_partition or {}
        self._build_task_index()

    def _build_task_index(self):
        """构建任务ID到类型的映射"""
        self.task_type_index = {}
        for task_type, tasks in self.task_partition.items():
            for task in tasks:
                self.task_type_index[task.get("id", task.get("task_id"))] = task_type

    def get_task_type(self, task_id: int) -> str:
        """获取任务类型"""
        return self.task_type_index.get(task_id, "unseen")

    def evaluate_task(
        self,
        task_id: int,
        trajectories: List,  # N 条轨迹
        n: int = 1,  # 计算 pass@n
    ) -> PassKResult:
        """
        评测单个任务
        pass@n = 至少有一条轨迹成功
        """
        task_type = self.get_task_type(task_id)

        # 检查是否有成功的轨迹
        success = any(traj.success for traj in trajectories[:n])

        # 统计（使用第一条轨迹的统计）
        first_traj = trajectories[0]
        num_turns = first_traj.num_turns
        num_tool_calls = first_traj.num_tool_calls
        error = first_traj.error

        return PassKResult(
            task_id=task_id,
            task_type=task_type,
            success=success,
            num_samples=len(trajectories),
            num_turns=num_turns,
            num_tool_calls=num_tool_calls,
            error=error,
        )

    def evaluate(
        self,
        results: Dict[int, List],  # task_id -> N 条轨迹
        n: int = 1,
    ) -> AggregatedPassKResult:
        """
        评测所有任务
        """
        per_task_results = []

        for task_id, trajectories in results.items():
            result = self.evaluate_task(task_id, trajectories, n)
            per_task_results.append(result)

        # 按类型聚合
        covered_results = [r for r in per_task_results if r.task_type == "covered_seen"]
        uncovered_results = [r for r in per_task_results if r.task_type == "uncovered_seen"]
        unseen_results = [r for r in per_task_results if r.task_type == "unseen"]

        overall_pass_at_1 = np.mean([r.success for r in per_task_results])
        covered_seen_pass_at_1 = np.mean([r.success for r in covered_results]) if covered_results else 0.0
        uncovered_seen_pass_at_1 = np.mean([r.success for r in uncovered_results]) if uncovered_results else 0.0
        unseen_pass_at_1 = np.mean([r.success for r in unseen_results]) if unseen_results else 0.0

        # 泛化指标：(uncovered_seen * 24 + unseen * 10) / 34
        # 排除训练集泄漏影响
        gen_score = (
            uncovered_seen_pass_at_1 * len(uncovered_results) +
            unseen_pass_at_1 * len(unseen_results)
        ) / max(1, len(uncovered_results) + len(unseen_results))

        return AggregatedPassKResult(
            overall_pass_at_1=overall_pass_at_1,
            covered_seen_pass_at_1=covered_seen_pass_at_1,
            uncovered_seen_pass_at_1=uncovered_seen_pass_at_1,
            unseen_pass_at_1=unseen_pass_at_1,
            generalization_pass_at_1=gen_score,
            total_tasks=len(per_task_results),
            per_task_results=per_task_results,
        )

    def save_results(self, results: AggregatedPassKResult, output_path: str):
        """保存评测结果"""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w", encoding="utf-8") as f:
            json.dump(results.to_dict(), f, ensure_ascii=False, indent=2)

    def print_summary(self, results: AggregatedPassKResult):
        """打印评测摘要"""
        print("\n" + "=" * 60)
        print("Pass@1 评测结果")
        print("=" * 60)
        print(f"总体 Pass@1:      {results.overall_pass_at_1:.3f}")
        print(f"Covered Seen:      {results.covered_seen_pass_at_1:.3f}")
        print(f"Uncovered Seen:    {results.uncovered_seen_pass_at_1:.3f}")
        print(f"Unseen:            {results.unseen_pass_at_1:.3f}")
        print(f"泛化 Pass@1:       {results.generalization_pass_at_1:.3f}")
        print(f"总任务数:          {results.total_tasks}")
        print("=" * 60)


def load_results_from_dir(results_dir: str) -> Dict[int, List]:
    """
    从目录加载评测结果
    目录结构：results_dir/{task_id}/sample_*.json
    """
    results = {}
    results_path = Path(results_dir)

    for task_dir in results_path.iterdir():
        if not task_dir.is_dir():
            continue

        task_id = int(task_dir.name)
        trajectories = []

        for sample_file in task_dir.glob("sample_*.json"):
            with open(sample_file, "r") as f:
                traj_data = json.load(f)
                # 转换为 TrajectoryResult
                from ..envs.tau_bench_wrapper import TrajectoryResult
                traj = TrajectoryResult(
                    task_id=traj_data["task_id"],
                    success=traj_data["success"],
                    reward=traj_data["reward"],
                    num_turns=traj_data["num_turns"],
                    num_tool_calls=traj_data["num_tool_calls"],
                    raw_messages=traj_data.get("raw_messages", []),
                    error=traj_data.get("error"),
                )
                trajectories.append(traj)

        results[task_id] = trajectories

    return results
