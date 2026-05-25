#!/usr/bin/env python3
"""
服务器环境检查脚本
在服务器上运行此脚本，验证训练环境是否准备好
"""
import os
import sys
import subprocess
from pathlib import Path


class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'


def check(description, command=None, path=None, min_version=None):
    """检查项"""
    print(f"检查 {description}...", end=" ")

    if command:
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version = result.stdout.strip().split('\n')[0]
                print(f"{Colors.GREEN}✓{Colors_NC} {version}")
                return True
            else:
                print(f"{Colors.RED}✗{Colors_NC}")
                return False
        except Exception as e:
            print(f"{Colors.RED}✗{Colors_NC} ({e})")
            return False

    elif path:
        p = Path(path)
        if p.exists():
            print(f"{Colors.GREEN}✓{Colors_NC}")
            return True
        else:
            print(f"{Colors.RED}✗{Colors_NC} (不存在)")
            return False

    return False


def main():
    print(f"{Colors.YELLOW}IG-GRPO 环境检查{Colors_NC}")
    print("=" * 50)
    print()

    results = []

    # 基础环境
    print(f"{Colors.YELLOW}基础环境{Colors_NC}")
    results.append(check("Python", "python --version"))
    results.append(check("CUDA", "nvidia-smi"))
    results.append(check("pip", "pip --version"))

    print()

    # GPU 检查
    print(f"{Colors.YELLOW}GPU 检查{Colors_NC}")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            gpus = result.stdout.strip().split('\n')
            print(f"找到 {len(gpus)} 个 GPU:")
            for i, gpu in enumerate(gpus, 1):
                print(f"  GPU {i}: {gpu}")
            results.append(True)
        else:
            print(f"{Colors.RED}无法获取 GPU 信息{Colors_NC}")
            results.append(False)
    except Exception as e:
        print(f"{Colors.RED}GPU 检查失败: {e}{Colors_NC}")
        results.append(False)

    print()

    # Python 包检查
    print(f"{Colors.YELLOW}Python 包检查{Colors_NC}")
    packages = [
        ("torch", "torch --version"),
        ("transformers", "transformers-cli --version"),
        ("vllm", "vllm --version"),
    ]

    for name, cmd in packages:
        results.append(check(name, cmd))

    print()

    # 文件检查
    print(f"{Colors.YELLOW}文件检查{Colors_NC}")
    files = [
        ("配置目录", "configs/train/grpo/ig_full_4090.yaml"),
        ("工具配置", "configs/tool_config/tau_bench_retail_tools.yaml"),
        ("交互配置", "configs/interaction_config/tau_bench_retail_jig.yaml"),
        ("源码目录", "src/envs"),
        ("脚本目录", "scripts/train/grpo"),
    ]

    for name, path in files:
        results.append(check(name, path=path))

    print()

    # 磁盘空间
    print(f"{Colors.YELLOW}磁盘空间{Colors_NC}")
    try:
        result = subprocess.run(
            ["df", "-h", "."],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                print(lines[0])
                print(lines[1])
                # 检查可用空间
                avail = lines[1].split()[3]
                print(f"可用空间: {avail}")
    except Exception as e:
        print(f"{Colors.RED}无法检查磁盘空间: {e}{Colors_NC}")

    print()

    # 端口检查
    print(f"{Colors.YELLOW}端口检查{Colors_NC}")
    ports = [8000, 8001]
    for port in ports:
        try:
            result = subprocess.run(
                ["lsof", "-i", f":{port}"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                print(f"端口 {port}: {Colors.YELLOW}已被占用{Colors_NC}")
                results.append(False)
            else:
                print(f"端口 {port}: {Colors.GREEN}可用{Colors_NC}")
                results.append(True)
        except Exception:
            # lsof 可能不存在，跳过
            print(f"端口 {port}: {Colors.YELLOW}未检查{Colors_NC}")

    print()
    print("=" * 50)

    # 总结
    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"总计: {total} 项, {Colors.GREEN}通过: {passed}{Colors_NC}, {Colors.RED}失败: {failed}{Colors_NC}")

    if failed == 0:
        print(f"{Colors.GREEN}✓ 环境准备完成，可以开始训练！{Colors_NC}")
        return 0
    else:
        print(f"{Colors.YELLOW}⚠ 有 {failed} 项检查失败，请修复后重试{Colors_NC}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
