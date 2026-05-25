#!/usr/bin/env python3
"""
聚合 Pass@K 评测结果
将多个检查点的评测结果聚合成一个表格
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    parser.add_argument("--output_file", type=str, default="pass_k_aggregated.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    results = {}
    for ckpt in args.checkpoints:
        result_file = output_dir / f"pass_k_{ckpt}.json"
        if result_file.exists():
            with open(result_file) as f:
                results[ckpt] = json.load(f)
        else:
            print(f"警告: 找不到 {result_file}")

    # 聚合
    aggregated = {
        "checkpoints": sorted(results.keys()),
        "overall_pass_at_1": [],
        "covered_seen_pass_at_1": [],
        "uncovered_seen_pass_at_1": [],
        "unseen_pass_at_1": [],
        "generalization_pass_at_1": [],
    }

    for ckpt in sorted(results.keys()):
        r = results[ckpt]
        aggregated["overall_pass_at_1"].append(r.get("overall_pass_at_1", 0))
        aggregated["covered_seen_pass_at_1"].append(r.get("covered_seen_pass_at_1", 0))
        aggregated["uncovered_seen_pass_at_1"].append(r.get("uncovered_seen_pass_at_1", 0))
        aggregated["unseen_pass_at_1"].append(r.get("unseen_pass_at_1", 0))
        aggregated["generalization_pass_at_1"].append(r.get("generalization_pass_at_1", 0))

    # 统计
    aggregated["mean_pass_at_1"] = np.mean(aggregated["overall_pass_at_1"])
    aggregated["max_pass_at_1"] = np.max(aggregated["overall_pass_at_1"])
    aggregated["best_checkpoint"] = aggregated["checkpoints"][np.argmax(aggregated["overall_pass_at_1"])]

    # 保存
    output_file = output_dir / args.output_file
    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"聚合结果已保存到: {output_file}")

    # 打印表格
    print("\n检查点 | Pass@1 | Covered | Uncovered | Unseen | 泛化")
    print("-" * 70)
    for ckpt in sorted(results.keys()):
        r = results[ckpt]
        print(f"{ckpt:6} | {r.get('overall_pass_at_1', 0):.3f} | {r.get('covered_seen_pass_at_1', 0):.3f} | {r.get('uncovered_seen_pass_at_1', 0):.3f} | {r.get('unseen_pass_at_1', 0):.3f} | {r.get('generalization_pass_at_1', 0):.3f}")

    print(f"\n最佳检查点: {aggregated['best_checkpoint']}")
    print(f"平均 Pass@1: {aggregated['mean_pass_at_1']:.3f}")
    print(f"最高 Pass@1: {aggregated['max_pass_at_1']:.3f}")


if __name__ == "__main__":
    main()
