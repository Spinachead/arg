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
    
    # 提取工具执行结果作为额外上下文
    tool_result_context = ""
    if is_after_tool and last_message:
        # 转义大括号，防止被当作模板变量解析 (特别是工具返回 JSON/dict 时)
        content = last_message.content
        safe_content = str(content).replace("{", "{{").replace("}", "}}")
        tool_result_context = f"\n[最新工具执行结果]\n这是你刚才调用工具返回的实时信息，请优先参考此内容：\n{safe_content}\n"
        print(f"this is tool_result_context: {tool_result_context}")
    
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
       # 1. 构造系统提示词（只放人设、工具指南和环境上下文）
    system_prompt = f"""你是一个强大且专业的智能助手。请基于提供的[已知信息]和[最新工具执行结果]来准确、简洁地回答用户问题。

[环境上下文]
- 当前用户 ID: {state.user_id}

[工具使用指南]
1. 当用户要求"记住"、"保存"某些信息时，调用相关的存储工具。
2. 当用户询问"之前说过什么"、"我记了什么"时，调用相关的查询工具。
3. 当需要访问外部数据时，优先考虑使用可用的工具。
4. 如果工具已经返回了结果，请务必将其视为最可信的实时数据来源。

请根据提供的背景信息（包括工具结果和检索到的文档）做出综合回答。
"""

    # 2. 获取标准的 RAG 提问模板 (包含 [已知信息] 和 [问题] 占位符)
    prompt_template = get_prompt_template("rag", "default")

    # 3. 构建完整的对话模板
    # 使用 History 类来确保使用 jinja2 引擎解析 {{context}} 和 {{question}}
    prompt_messages = [("system", system_prompt)]
    
    # 遍历历史消息（排除掉最后一条，因为最后一条要用 RAG 模板包装）
    for msg in state.messages[:-1]:
        h = History(role=msg.type, content=msg.content)
        # is_raw=True 会用 {% raw %} 包装内容，防止历史消息中的特殊字符干扰解析
        prompt_messages.append(h.to_msg_template(is_raw=True))
    
    # 最后一条：使用 RAG 模板，注意 is_raw=False，这样才能解析模板里的变量
    h_rag = History(role="user", content=prompt_template)
    prompt_messages.append(h_rag.to_msg_template(is_raw=False))

    chat_prompt = ChatPromptTemplate.from_messages(prompt_messages)

    # 4. 准备合并后的 Context（工具结果 + 检索文档）
    final_context = state.context
    if tool_result_context:
        final_context = f"{tool_result_context}\n\n[检索到的参考文档]\n{state.context}\n[来源]: {state.sources}"

    # 5. 调用 LLM
    chain = chat_prompt | model
    response = await chain.ainvoke({
        "context": final_context,
        "question": state.query if state.query else state.messages[-1].content,
    }, config)
    
    return {"messages": [response]}