from core.state_graph.states.main_graph.agent_state import AgentState
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage
from config import config as app_config
from utils import get_prompt_template, History
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
import chainlit as cl


async def respond(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """调用 LLM 生成回复，支持 MCP 工具调用"""
    
    # 初始化模型
    model = init_chat_model(name="respond", **app_config["inference_model_params"])
    
    # 检查上一条消息是否是工具执行结果（防止无限循环）
    last_message = state.messages[-1] if state.messages else None
    is_after_tool = isinstance(last_message, ToolMessage)
    
    # 获取 MCP 工具并绑定到模型（工具执行后不再绑定）
    if not is_after_tool:
        mcp_tools_dict = cl.user_session.get("mcp_tools", {})
        mcp_tools = []
        for tools in mcp_tools_dict.values():
            mcp_tools.extend(tools)
        
        if mcp_tools:
            model = model.bind_tools(mcp_tools)
    
    # 构建提示词
    prompt_template = get_prompt_template("rag", "default")
    system_prompt = f"""{prompt_template}
    
[环境上下文]
- 当前用户 ID: {state.user_id}

[工具使用指南]
你可以使用以下工具来增强回答能力：
1. 当用户要求"记住"、"保存"某些信息时，调用相关的存储工具
2. 当用户询问"之前说过什么"、"我记了什么"时，调用相关的查询工具
3. 当需要访问外部数据或执行特定操作时，优先考虑使用可用的工具
4. 如果用户的问题明确需要工具才能完成，请主动调用工具

请根据工具的描述，在必要时主动调用它们以获取准确信息。
"""
    
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        History(role="user", content=prompt_template).to_msg_template(False)
    ])

    # 调用 LLM
    chain = chat_prompt | model
    response = await chain.ainvoke({
        "context": state.context,
        "sources": state.sources if state.sources else "未知来源",
        "question": state.query if state.query else state.messages[-1].content,
    }, config)
    
    return {"messages": [response]}