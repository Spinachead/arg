from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Union

# from langchain.agents import create_agent  # 在 langchain 1.0 中，create_agent 可能已移动或改名
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from pydantic.v1 import BaseModel, Field

from langchain_chatchat.utils.try_parse_json_object import try_parse_json_object

logger = logging.getLogger(__name__)


@tool
def generic_tool(query: str) -> str:
    """
    通用的查询工具 - 替换为你的实际工具
    query: 查询参数 (支持 JSON 或字符串)
    """
    # 这里处理你的实际工具逻辑
    try:
        # 如果是 JSON，解析它
        if validate_json(query):
            data = json.loads(query)
        else:
            data = {"query": query}
        logger.info(f"工具执行: {data}")
        return f"工具执行结果: {query}"
    except Exception as e:
        return f"工具执行失败: {str(e)}"


class QwenFinalAnswer(BaseModel):
    """结构化最终答案"""
    output: str = Field(description="最终答案内容")
    confidence: float = Field(default=1.0, description="置信度 0-1")


def create_qwen_agent_v1(model, tools: List = None):
    """
    v1.0 Qwen Agent - 无需自定义解析器！

    Args:
        model: Qwen 模型实例
        tools: 工具列表 (默认使用 generic_tool)
    """
    if tools is None:
        tools = [generic_tool]

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt="""你是一个智能助手，使用以下格式响应：

工具调用格式：
- 识别用户意图，调用合适工具
- 输入可以是 JSON 或普通文本

最终答案格式：
- 当不需要工具时，直接给出答案

示例：
用户问天气 → 调用天气工具
用户问数学题 → 调用计算工具
其他问题 → Final Answer""",
        response_format=QwenFinalAnswer  # 自动结构化最终答案
    )
    return agent


def validate_json(json_data: str) -> bool:
    """验证 JSON 格式"""
    try:
        json.loads(json_data)
        return True
    except ValueError:
        return False


def preprocess_qwen_output(text: str) -> str:
    """
    可选：预处理 Qwen 特殊输出格式（如果模型仍使用旧格式）
    v1.0 通常不需要，但保留以兼容
    """
    # 匹配 Action 和 Action Input
    if match := re.search(r"\nAction:\s*(.+)\nAction\sInput:\s*(.+)", text, re.DOTALL):
        action_name = match.group(1).strip()
        action_input = match.group(2).strip()

        # 解析工具输入
        _, json_input = try_parse_json_object(action_input)

        # 转换为工具调用提示
        tool_json = {
            "tool": action_name,
            "tool_input": json_input or action_input
        }
        return f"请调用工具: {json.dumps(tool_json, ensure_ascii=False)}"

    # 匹配 Final Answer
    elif match := re.search(r"\nFinal\sAnswer:\s*(.+)", text, re.DOTALL):
        final_answer = match.group(1).strip()
        return f"Final Answer: {final_answer}"

    return text


# 使用示例
def run_qwen_agent(model, user_input: str):
    """运行 Qwen Agent"""
    agent = create_qwen_agent_v1(model)

    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # 自动解析结果
    final_message = result["messages"][-1]

    if hasattr(final_message, 'structured_response') and final_message.structured_response:
        # 结构化最终答案
        return final_message.structured_response.output
    elif final_message.tool_calls:
        # 工具调用已自动处理
        return "工具已调用，请查看执行结果"
    else:
        # 普通文本响应
        return final_message.content




def qwen_output_middleware():
    """Qwen 专用输出处理中间件（可选）"""

    def after_model(response, state):
        """处理模型原始输出"""
        if isinstance(response, AIMessage) and response.content:
            content = preprocess_qwen_output(response.content)
            response.content = content
        return response

