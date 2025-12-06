"""
LangChain v1.0 版本 - ChatGLM3-6B Agent
无需自定义解析器，使用 create_agent 自动处理
"""

import json
import re
from typing import Dict, Any
# from pydantic_v1 import BaseModel, Field

# from langchain.agents import create_agent  # 在 langchain 1.0 中，create_agent 可能已移动或改名
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from pydantic.v1 import BaseModel, Field

from langchain_chatchat.utils.try_parse_json_object import try_parse_json_object

# 示例工具 - 替换为你的实际工具
@tool
def execute_python_code(code: str) -> str:
    """执行 Python 代码"""
    try:
        exec(code)
        return "代码执行成功"
    except Exception as e:
        return f"代码执行失败: {str(e)}"

# 定义结构化输出格式（可选，用于 Final Answer）
class FinalAnswer(BaseModel):
    answer: str = Field(description="最终答案")
    confidence: float = Field(description="置信度 0-1")

# 创建 Agent - 无需解析器！
def create_glm3_agent(model):
    """
    创建 ChatGLM3 Agent
    model: 你的 ChatGLM3 模型实例
    """
    agent = create_agent(
        model=model,
        tools=[execute_python_code],  # 添加你的工具
        system_prompt="""你是一个代码助手。
        
识别用户需求：
1. 如果需要执行代码，使用 execute_python_code 工具
2. 其他情况直接给出 Final Answer

输出格式：
- 工具调用：直接调用工具
- 最终答案：使用结构化输出""",
        response_format=FinalAnswer  # 可选：结构化最终答案
    )
    return agent

# 使用示例
def run_agent(model, user_input: str):
    agent = create_glm3_agent(model)

    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # 自动解析，无需手动处理！
    final_msg = result["messages"][-1]

    if hasattr(final_msg, 'structured_response'):
        return final_msg.structured_response.answer
    else:
        return final_msg.content


def glm3_output_middleware():
    """处理 ChatGLM3 特殊输出格式的中介件"""

    def wrap_model_call(request, handler):
        # ChatGLM3 可能返回特殊格式，预处理
        messages = request.messages

        # 处理包含 ```python tool_call()``` 的特殊格式
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage) and "```python" in msg.content:
                # 提取并转换为标准工具调用格式
                content = preprocess_glm3_output(msg.content)
                messages[i] = HumanMessage(content=content)

        result = handler(request)
        return result



def preprocess_glm3_output(text: str) -> str:
    """预处理 ChatGLM3 的特殊输出格式"""
    # 保留原有的正则匹配逻辑
    if match := re.search(r'(\S+\s+```python\s+tool_call\(.*?\)\s+```)', text, re.DOTALL):
        exec_code = match.group(1)
        action = exec_code.split("```python")[0].replace("\n", "").strip()
        code_str = "```" + exec_code.split("```python")[1]

        _, params = try_parse_json_object(code_str)
        action_json = {"action": action, "action_input": params}

        return f"Action: ```{json.dumps(action_json, ensure_ascii=False)}```"

    return text