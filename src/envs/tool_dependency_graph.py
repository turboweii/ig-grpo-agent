"""
工具依赖图 - IG-GRPO 可选组件
建模工具之间的依赖/互补关系，奖励符合依赖关系的工具调用
"""
from __future__ import annotations
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import json


class ToolDependencyGraph:
    """
    工具依赖图
    基于规则/工具定义构建依赖关系，奖励合理的工具序列
    """

    def __init__(self, domain: str = "retail"):
        self.domain = domain
        self.graph: Dict[str, Set[str]] = defaultdict(set)  # tool -> 依赖的工具
        self.complementary: Dict[Tuple[str, str], float] = {}  # (tool_i, tool_j) -> 互补度

        # 学习到的转移概率
        self.learned_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        self.total_transitions = 0

        # 构建领域特定的依赖图
        self._build_domain_graph(domain)

    def _build_domain_graph(self, domain: str):
        """基于领域知识构建依赖图"""
        if domain == "retail":
            self._build_retail_graph()
        elif domain == "airline":
            self._build_airline_graph()
        elif domain == "hotel":
            self._build_hotel_graph()

    def _build_retail_graph(self):
        """
        Retail 领域的依赖关系
        典型流程：搜索 -> 查看详情 -> 添加到购物车 -> 结账
        """
        dependencies = {
            # 搜索类工具
            "Products.search": [],
            "Products.get_by_id": ["Products.search"],  # 需要 search 提供的产品 ID

            # 购物车类
            "Cart.add": ["Products.get_by_id"],  # 需要先查看产品
            "Cart.remove": ["Cart.add"],
            "Cart.modify": ["Cart.add"],

            # 订单类
            "Orders.create": ["Cart.add"],  # 需要先加购物车
            "Orders.modify": ["Orders.create"],
            "Orders.cancel": ["Orders.create"],

            # 用户类
            "User.get_profile": [],
            "User.update_profile": ["User.get_profile"],

            # 退换货
            "Returns.request": ["Orders.create"],
            "Returns.check_status": ["Returns.request"],
        }

        # 互补关系（经常一起使用）
        complementarities = {
            ("Products.search", "Products.get_by_id"): 0.3,
            ("Cart.add", "Cart.view"): 0.2,
            ("Orders.create", "Orders.view"): 0.3,
            ("Products.search", "Products.search"): 0.1,  # 允许重新搜索
        }

        for tool, deps in dependencies.items():
            self.graph[tool] = set(deps)

        for (t1, t2), score in complementarities.items():
            self.complementary[(t1, t2)] = score

    def _build_airline_graph(self):
        """Airline 领域的依赖关系"""
        dependencies = {
            "search_direct_flight": [],
            "search_onestop_flight": [],
            "book_reservation": ["search_direct_flight", "search_onestop_flight"],
            "modify_reservation": ["book_reservation"],
            "cancel_reservation": ["book_reservation"],
            "list_all_airports": [],
            "get_reservation_details": ["book_reservation"],
        }

        complementarities = {
            ("search_direct_flight", "search_onestop_flight"): 0.2,
            ("search_direct_flight", "book_reservation"): 0.3,
            ("book_reservation", "get_reservation_details"): 0.2,
        }

        for tool, deps in dependencies.items():
            self.graph[tool] = set(deps)

        for (t1, t2), score in complementarities.items():
            self.complementary[(t1, t2)] = score

    def _build_hotel_graph(self):
        """Hotel 领域的依赖关系"""
        dependencies = {
            "Hotels.search": [],
            "Hotels.get_details": ["Hotels.search"],
            "Bookings.create": ["Hotels.get_details"],
            "Bookings.modify": ["Bookings.create"],
            "Bookings.cancel": ["Bookings.create"],
        }

        complementarities = {
            ("Hotels.search", "Hotels.get_details"): 0.3,
            ("Hotels.get_details", "Bookings.create"): 0.3,
        }

        for tool, deps in dependencies.items():
            self.graph[tool] = set(deps)

        for (t1, t2), score in complementarities.items():
            self.complementary[(t1, t2)] = score

    def get_dependency_reward(self, prev_tool: str, curr_tool: str) -> float:
        """
        计算依赖奖励
        如果 curr_tool 依赖于 prev_tool → 高奖励
        """
        # 检查直接依赖
        if curr_tool in self.graph:
            dependencies = self.graph[curr_tool]
            if prev_tool in dependencies:
                return 0.3  # 直接依赖，高奖励

        # 检查互补关系
        key = (prev_tool, curr_tool)
        if key in self.complementary:
            return self.complementary[key]

        # 检查反向互补
        key_reverse = (curr_tool, prev_tool)
        if key_reverse in self.complementary:
            return self.complementary[key_reverse] * 0.5

        return 0.0

    def update_transition(self, prev_tool: str, curr_tool: str):
        """更新学习到的转移计数"""
        self.learned_transitions[(prev_tool, curr_tool)] += 1
        self.total_transitions += 1

    def get_learned_probability(self, tool_i: str, tool_j: str) -> float:
        """获取学习到的转移概率"""
        if self.total_transitions == 0:
            return 0.0
        return self.learned_transitions.get((tool_i, tool_j), 0) / self.total_transitions

    def get_suggested_tools(self, current_tool: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        获取建议的下一个工具
        基于依赖关系和学习到的转移概率
        """
        suggestions = []

        # 基于依赖关系
        for tool, deps in self.graph.items():
            if current_tool in deps:
                suggestions.append((tool, 0.5))  # 高优先级

        # 基于互补关系
        for (t1, t2), score in self.complementary.items():
            if t1 == current_tool:
                suggestions.append((t2, score))
            elif t2 == current_tool:
                suggestions.append((t1, score * 0.5))

        # 排序并返回 top-k
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:top_k]

    def visualize(self) -> str:
        """生成依赖图的文本表示"""
        lines = ["Tool Dependency Graph:"]
        for tool, deps in sorted(self.graph.items()):
            if deps:
                lines.append(f"  {tool} -> {', '.join(sorted(deps))}")
            else:
                lines.append(f"  {tool} (no dependencies)")

        if self.complementary:
            lines.append("\nComplementary Relationships:")
            for (t1, t2), score in sorted(self.complementary.items()):
                lines.append(f"  {t1} <-> {t2}: {score:.2f}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """导出为字典（用于保存/加载）"""
        return {
            "domain": self.domain,
            "graph": {k: list(v) for k, v in self.graph.items()},
            "complementary": {f"{k[0]}->{k[1]}": v for k, v in self.complementary.items()},
            "learned_transitions": {f"{k[0]}->{k[1]}": v for k, v in self.learned_transitions.items()},
            "total_transitions": self.total_transitions,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolDependencyGraph":
        """从字典恢复"""
        graph = cls(domain=data.get("domain", "retail"))

        # 恢复 graph
        for tool, deps in data.get("graph", {}).items():
            graph.graph[tool] = set(deps)

        # 恢复 complementary
        for key, score in data.get("complementary", {}).items():
            parts = key.split("->")
            if len(parts) == 2:
                graph.complementary[(parts[0], parts[1])] = score

        # 恢复 learned_transitions
        for key, count in data.get("learned_transitions", {}).items():
            parts = key.split("->")
            if len(parts) == 2:
                graph.learned_transitions[(parts[0], parts[1])] = count

        graph.total_transitions = data.get("total_transitions", 0)

        return graph
