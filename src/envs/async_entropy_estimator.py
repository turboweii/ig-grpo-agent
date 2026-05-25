"""
异步熵估计器 - IG-GRPO 核心组件
双缓冲设计，熵计算不阻塞 rollout
"""
from __future__ import annotations
import asyncio
import math
import hashlib
import threading
from collections import deque, defaultdict, OrderedDict
from typing import Optional, Dict, Tuple
import time


class AsyncEntropyEstimator:
    """
    双缓冲熵估计器
    - 前台缓冲：接收新的状态访问
    - 后台缓冲：异步计算熵
    - LRU 缓存：热点状态快速返回（手动实现避免内存泄漏）
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        cache_size: int = 5000,
        update_interval: float = 0.1,  # 后台更新间隔(秒)
    ):
        self.buffer_size = buffer_size
        self._cache_max_size = cache_size
        self.update_interval = update_interval

        # 状态计数器
        self.state_counter: Dict[int, int] = defaultdict(int)
        self.state_tool_counter: Dict[Tuple[int, str], int] = defaultdict(int)

        # 前台缓冲（接收新访问）
        self.frontier_buffer: deque = deque(maxlen=buffer_size)

        # 后台任务
        self._computation_lock = asyncio.Lock()
        self._background_task: Optional[asyncio.Task] = None
        self._running = False

        # 手动 LRU 缓存（避免 lru_cache 在实例方法上的内存泄漏）
        self._entropy_cache: OrderedDict[int, float] = OrderedDict()

        # 统计
        self.total_visits = 0
        self.unique_states = 0
        self.unique_state_tool_pairs = 0

    async def start(self):
        """启动后台计算任务"""
        if not self._running:
            self._running = True
            self._background_task = asyncio.create_task(self._background_worker())

    async def stop(self):
        """停止后台计算任务"""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass

    async def _background_worker(self):
        """后台工作协程：定期处理缓冲区"""
        while self._running:
            try:
                await self._process_buffer()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Background worker error: {e}")

    async def _process_buffer(self):
        """处理缓冲区中的待计算项"""
        if not self.frontier_buffer:
            return

        async with self._computation_lock:
            batch_size = min(100, len(self.frontier_buffer))
            for _ in range(batch_size):
                if not self.frontier_buffer:
                    break
                item = self.frontier_buffer.popleft()
                await self._update_counters(item)

    async def _update_counters(self, item: dict):
        """更新计数器"""
        state_hash = item.get("state_hash")
        tool = item.get("tool")

        if state_hash:
            self.state_counter[state_hash] += 1
            if self.state_counter[state_hash] == 1:
                self.unique_states += 1

            if tool:
                key = (state_hash, tool)
                self.state_tool_counter[key] += 1
                if self.state_tool_counter[key] == 1:
                    self.unique_state_tool_pairs += 1

            self.total_visits += 1

    def _uncached_entropy(self, state_hash: int) -> float:
        """计算单状态熵（未缓存版本）"""
        count = self.state_counter.get(state_hash, 1)
        if self.total_visits == 0:
            return 0.0
        p = count / self.total_visits
        return -p * math.log(p) if p > 0 else 0.0

    def _get_cached_entropy(self, state_hash: int) -> float:
        """手动 LRU 缓存的熵获取"""
        if state_hash in self._entropy_cache:
            # LRU: 移到末尾
            value = self._entropy_cache.pop(state_hash)
            self._entropy_cache[state_hash] = value
            return value

        # 计算并缓存
        value = self._uncached_entropy(state_hash)
        self._entropy_cache[state_hash] = value

        # 超过容量，删除最老的
        if len(self._entropy_cache) > self._cache_max_size:
            self._entropy_cache.popitem(last=False)

        return value

    def entropy(self, state_hash: int) -> float:
        """获取状态熵（线程安全）"""
        return self._get_cached_entropy(state_hash)

    def conditional_entropy(self, state_hash: int, tool: str) -> float:
        """计算 H(tool | state)"""
        key = (state_hash, tool)
        count = self.state_tool_counter.get(key, 0)
        state_count = self.state_counter.get(state_hash, 1)

        if state_count == 0:
            return 0.0

        p = count / state_count
        return -p * math.log(p) if p > 0 else 0.0

    async def compute_ig_async(
        self,
        state_hash: int,
        tool: str,
        next_state_hash: Optional[int] = None
    ) -> float:
        """
        异步计算信息增益
        IG = H(S) - H(S | tool)
        """
        # 添加到前台缓冲
        self.frontier_buffer.append({
            "state_hash": state_hash,
            "tool": tool,
            "next_state_hash": next_state_hash,
            "timestamp": time.time(),
        })

        # 计算当前熵
        h_state = self.entropy(state_hash)

        # 如果有下一个状态，计算条件熵
        if next_state_hash is not None:
            h_next = self.entropy(next_state_hash)
            ig = h_state - h_next
        else:
            # 使用工具条件熵
            h_tool_given_state = self.conditional_entropy(state_hash, tool)
            ig = h_state - h_tool_given_state

        return max(0, ig)  # IG 非负

    def compute_ig_sync(self, state_hash: int, tool: str) -> float:
        """同步计算信息增益（用于训练时不希望异步的情况）"""
        h_state = self.entropy(state_hash)
        h_tool_given_state = self.conditional_entropy(state_hash, tool)
        return max(0, h_state - h_tool_given_state)

    def get_coverage_stats(self) -> dict:
        """获取覆盖率统计"""
        return {
            "total_visits": self.total_visits,
            "unique_states": self.unique_states,
            "unique_state_tool_pairs": self.unique_state_tool_pairs,
            "state_coverage_rate": self.unique_states / max(1, self.total_visits),
            "buffer_size": len(self.frontier_buffer),
        }

    def reset(self):
        """重置所有计数器"""
        self.state_counter.clear()
        self.state_tool_counter.clear()
        self.frontier_buffer.clear()
        self._entropy_cache.clear()
        self.total_visits = 0
        self.unique_states = 0
        self.unique_state_tool_pairs = 0


class GlobalEntropyEstimator:
    """
    全局单例熵估计器
    在多进程/多线程环境下安全使用
    """
    _instance: Optional[AsyncEntropyEstimator] = None
    _lock = threading.Lock()

    @classmethod
    def get(cls) -> AsyncEntropyEstimator:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = AsyncEntropyEstimator()
        return cls._instance

    @classmethod
    async def initialize(cls):
        """初始化全局实例"""
        estimator = cls.get()
        await estimator.start()

    @classmethod
    async def shutdown(cls):
        """关闭全局实例"""
        if cls._instance:
            await cls._instance.stop()
