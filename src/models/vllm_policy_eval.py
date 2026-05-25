"""
vLLM 策略包装器 - IG-GRPO
支持 IG 奖励计算的策略模型
"""
from __future__ import annotations
import json
import time
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from transformers import AutoTokenizer


@dataclass
class PolicyOutput:
    """策略输出"""
    message: dict
    logprobs: Optional[dict] = None
    hidden_states: Optional[List] = None  # IG-GRPO: 用于 bypass 模式


class VLLMPolicy:
    """
    vLLM 策略包装器
    连接到本地 vLLM 服务器，提供工具调用能力
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model_name: str = "qwen2.5-7b-instruct",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        tools: Optional[List[dict]] = None,
    ):
        if OpenAI is None:
            raise ImportError("openai package is required for VLLMPolicy")

        self.client = OpenAI(
            base_url=base_url,
            api_key="dummy",  # vLLM 不需要真实的 API key
        )
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools or []

        # 加载 tokenizer（用于工具名称验证）
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name.replace("-vllm", ""), trust_remote_code=True
            )
        except Exception:
            self.tokenizer = None

        # IG-GRPO: 状态追踪
        self.was_truncated = False
        self.current_state_hash = None

    def set_tools(self, tools: List[dict]):
        """设置可用工具列表"""
        self.tools = tools

    def _format_tools(self) -> str:
        """将工具列表转换为系统提示格式"""
        if not self.tools:
            return ""

        tool_descs = []
        for tool in self.tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            param_str = json.dumps(params, ensure_ascii=False)
            tool_descs.append(f"- {name}: {desc}\n  参数: {param_str}")

        return "\n".join(tool_descs)

    def _build_messages(self, messages: List[dict]) -> List[dict]:
        """构建消息列表，添加工具说明"""
        if not self.tools:
            return messages

        # 检查是否已有 system 消息
        has_system = any(msg.get("role") == "system" for msg in messages)

        tool_instruction = f"\n# 可用工具\n{self._format_tools()}\n\n"
        tool_instruction += "工具调用格式：\n"
        tool_instruction += '请在回复中使用以下格式调用工具：\n'
        tool_instruction += '```json\n{{"tool_calls": [{{"function": {{"name": "工具名", "arguments": {{}}}}]}}}\n```'

        if has_system:
            # 在现有 system 消息后添加
            result = []
            for msg in messages:
                if msg.get("role") == "system" and "可用工具" not in msg.get("content", ""):
                    result.append({
                        "role": "system",
                        "content": msg.get("content", "") + tool_instruction,
                    })
                else:
                    result.append(msg)
            return result
        else:
            # 插入新的 system 消息
            return [{"role": "system", "content": tool_instruction}] + messages

    def __call__(self, messages: List[dict]) -> dict:
        """执行策略推理"""
        formatted_messages = self._build_messages(messages)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # 解析响应
            assistant_msg = {
                "role": "assistant",
                "content": response.choices[0].message.content or "",
            }

            # 检查是否有工具调用
            tool_calls = response.choices[0].message.tool_calls
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ]

            # IG-GRPO: vLLM 可能返回 logprobs（如果启用）
            if hasattr(response.choices[0], "logprobs") and response.choices[0].logprobs:
                assistant_msg["logprobs"] = response.choices[0].logprobs

            return assistant_msg

        except Exception as e:
            # 降级：返回错误消息
            return {
                "role": "assistant",
                "content": f"Error: {str(e)}",
            }

    def generate_with_logprobs(
        self,
        messages: List[dict],
        sample_size: int = 1,
    ) -> List[PolicyOutput]:
        """
        生成多个样本并返回 logprobs
        用于 IG-GRPO 的 group rollout
        """
        outputs = []

        for _ in range(sample_size):
            formatted_messages = self._build_messages(messages)

            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=formatted_messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    n=1,
                )

                choice = response.choices[0]

                assistant_msg = {
                    "role": "assistant",
                    "content": choice.message.content or "",
                }

                if choice.message.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in choice.message.tool_calls
                    ]

                output = PolicyOutput(
                    message=assistant_msg,
                    logprobs=getattr(choice, "logprobs", None),
                )

                outputs.append(output)

            except Exception as e:
                # 错误处理
                outputs.append(PolicyOutput(
                    message={
                        "role": "assistant",
                        "content": f"Error: {str(e)}",
                    }
                ))

        return outputs


class VLLMPolicyWithIG(VLLMPolicy):
    """
    支持 IG 奖励计算的策略包装器
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # IG 相关配置
        self.ig_weight = kwargs.get("ig_weight", 0.5)
        self.use_curriculum = kwargs.get("use_curriculum", False)
        self.curriculum_step = kwargs.get("curriculum_step", 0)
        self.curriculum_total_steps = kwargs.get("curriculum_total_steps", 300)

        # 组件引用（在外部设置）
        self.jig_computer = None
        self.entropy_estimator = None

    def get_current_ig_weight(self) -> float:
        """获取当前 IG 权重（课程学习）"""
        if not self.use_curriculum:
            return self.ig_weight

        # 课程学习：α(t) = α_0 * (1 - t/T)^β
        alpha_0 = self.ig_weight
        beta = 0.7
        t = self.curriculum_step
        T = self.curriculum_total_steps

        if t >= T:
            return 0.1  # 保持最小权重

        return alpha_0 * ((1 - t / T) ** beta)

    def set_jig_computer(self, jig_computer):
        """设置 JIG 计算器"""
        self.jig_computer = jig_computer

    def set_entropy_estimator(self, entropy_estimator):
        """设置熵估计器"""
        self.entropy_estimator = entropy_estimator

    def compute_ig_reward(
        self,
        state_hash: int,
        tool: str,
        prev_tool: Optional[str] = None,
    ) -> float:
        """计算 IG 奖励"""
        ig_weight = self.get_current_ig_weight()

        if self.jig_computer:
            # 使用 JIG 计算器
            jig = self.jig_computer.compute_jig(state_hash, tool, prev_tool)
            return ig_weight * jig

        elif self.entropy_estimator:
            # 使用熵估计器
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                ig = loop.run_until_complete(
                    self.entropy_estimator.compute_ig_async(state_hash, tool)
                )
            except RuntimeError:
                # 没有运行中的事件循环
                ig = self.entropy_estimator.compute_ig_sync(state_hash, tool)

            return ig_weight * ig

        else:
            return 0.0
