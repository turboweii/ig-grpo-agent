#!/usr/bin/env python3
"""
聚合 Entropy@K 评测结果
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
    parser.add_argument("--output_file", type=str, default="entropy_aggregated.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    results = {}
    for ckpt in args.checkpoints:
        result_file = output_dir / f"entropy_k_{ckpt}.json"
        if result_file.exists():
            with open(result_file) as f:
                results[ckpt] = json.load(f)
        else:
            print(f"警告: 找不到 {result_file}")

    # 聚合
    aggregated = {
        "checkpoints": sorted(results.keys()),
        "overall_success_rate": [],
        "overall_state_coverage": [],
        "overall_tool_entropy": [],
        "overall_path_diversity": [],
    }

    for ckpt in sorted(results.keys()):
        r = results[ckpt]
        aggregated["overall_success_rate"].append(r.get("overall_success_rate", 0))
        aggregated["overall_state_coverage"].append(r.get("overall_state_coverage", 0))
        aggregated["overall_tool_entropy"].append(r.get("overall_tool_entropy", 0))
        aggregated["overall_path_diversity"].append(r.get("overall_path_diversity", 0))

    # 统计
    aggregated["mean_success_rate"] = np.mean(aggregated["overall_success_rate"])
    aggregated["mean_state_coverage"] = np.mean(aggregated["overall_state_coverage"])
    aggregated["mean_tool_entropy"] = np.mean(aggregated["overall_tool_entropy"])
    aggregated["mean_path_diversity"] = np.mean(aggregated["overall_path_diversity"])

    # 保存
    output_file = output_dir / args.output_file
    with open(output_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"聚合结果已保存到: {output_file}")

    # 打印表格
    print("\n检查点 | Pass@1 | 覆盖率 | 工具熵 | 路径多样性")
    print("-" * 60)
    for ckpt in sorted(results.keys()):
        r = results[ckpt]
        print(f"{ckpt:6} | {r.get('overall_success_rate', 0):.3f} | {r.get('overall_state_coverage', 0):.3f} | {r.get('overall_tool_entropy', 0):.3f} | {r.get('overall_path_diversity', 0):.3f}")


if __name__ == "__main__":
    main()
