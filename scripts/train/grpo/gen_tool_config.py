#!/usr/bin/env python3
"""
生成 τ-bench 工具配置
用于 veRL rollout 和训练
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", type=str, default="retail", choices=["airline", "retail", "hotel"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # 从 τ-bench 导入工具定义
    from tau_bench.envs import get_env

    env = get_env(env_name=args.domain, task_split="train", task_index=0)

    # 转换工具格式
    tools = []
    for tool_info in env.tools_info:
        tool_def = {
            "type": "function",
            "function": {
                "name": tool_info["name"],
                "description": tool_info["description"],
                "parameters": tool_info.get("parameters", {
                    "type": "object",
                    "properties": {},
                }),
            },
        }
        tools.append(tool_def)

    # 输出
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(f"configs/tool_config/{args.domain}_tools.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(tools, f, indent=2, ensure_ascii=False)

    print(f"工具配置已生成: {output_path}")
    print(f"共 {len(tools)} 个工具")


if __name__ == "__main__":
    main()
