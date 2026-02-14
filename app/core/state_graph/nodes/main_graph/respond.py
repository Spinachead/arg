from core.state_graph.states.main_graph.agent_state import AgentState
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from settings import Settings
from utils import get_prompt_template, History
from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
import chainlit as cl
from core.state_graph.nodes.main_graph.tools import GENERAL_TOOLS


def convert_messages_for_model(messages: List) -> List:
    """
    将消息列表转换为模型可用的格式。
    特别处理 ToolMessage，确保保留 tool_call_id。
    """
    converted = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            # ToolMessage 需要保持原样，保留 tool_call_id
            converted.append(msg)
        elif isinstance(msg, AIMessage):
            # AIMessage 可能包含 tool_calls，也需要保持原样
            converted.append(msg)
        elif isinstance(msg, HumanMessage):
            converted.append(msg)
        else:
            # 其他消息类型，尝试转换
            converted.append(msg)
    return converted


async def respond(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """调用 LLM 生成回复，支持 MCP 工具调用"""
    
    # 获取用户自定义的模型配置
    user_settings = cl.user_session.get("model_settings", {})
    
    # 准备模型参数，优先使用用户设置，否则使用默认配置
    model_params = {
        "model": Settings.app_settings.inference_model,
        "temperature": Settings.app_settings.temperature,
        "streaming": Settings.app_settings.streaming,
        "openai_api_base": Settings.app_settings.openai_api_base,
        "openai_api_key": Settings.app_settings.openai_api_key,
    }
    
    if user_settings:
        # 更新模型参数
        if "model" in user_settings and user_settings["model"]:
            model_params["model"] = user_settings["model"]
        if "temperature" in user_settings:
            model_params["temperature"] = user_settings["temperature"]
        if "streaming" in user_settings:
            model_params["streaming"] = user_settings["streaming"]
        if "api_base" in user_settings and user_settings["api_base"]:
            model_params["openai_api_base"] = user_settings["api_base"]
        if "api_key" in user_settings and user_settings["api_key"]:
            model_params["openai_api_key"] = user_settings["api_key"]
    
    # 初始化模型
    model = init_chat_model(name="respond", **model_params)
    
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
    
    # 获取工具并绑定到模型（工具执行后不再绑定）
    if not is_after_tool:
        # 1. 获取 MCP 工具
        mcp_tools_dict = cl.user_session.get("mcp_tools", {})
        mcp_tools = []
        for tools in mcp_tools_dict.values():
            mcp_tools.extend(tools)
        
        # 2. 合并所有工具并绑定
        all_tools = mcp_tools + GENERAL_TOOLS
        if all_tools:
            model = model.bind_tools(all_tools)
    
    # 1. 构造系统提示词（只放人设、工具指南和环境上下文）
    system_prompt = f"""你是一个强大且专业的智能助手。请结合对话历史、已知信息和工具执行结果来准确、简洁地回答用户问题。

[环境上下文]
- 当前用户 ID: {state.user_id}

[回答原则]
1. 优先参考对话历史：如果用户问题涉及之前的对话内容，请基于对话历史回答
2. 结合已知信息：如果检索到的文档中有相关内容，也请一并参考
3. 使用工具：当需要访问外部数据时，优先考虑使用可用的工具

[工具使用指南]
1. 当用户要求"记住"、"保存"某些信息时，调用相关的存储工具。
2. 当用户询问"之前说过什么"、"我记了什么"时，调用相关的查询工具。
3. 如果工具已经返回了结果，请务必将其视为最可信的实时数据来源。
"""

    # 2. 获取标准的 RAG 提问模板 (包含 [已知信息] 和 [问题] 占位符)
    prompt_template = get_prompt_template("rag", "default")

    # 3. 构建完整的对话模板
    # 使用 History 类来确保使用 jinja2 引擎解析 {{context}} 和 {{question}}
    prompt_messages = [("system", system_prompt)]
    
    # 遍历历史消息（排除掉最后一条，因为最后一条要用 RAG 模板包装）
    for msg in state.messages[:-1]:
        # 特殊处理 ToolMessage：直接使用原消息，不通过 History 转换
        if isinstance(msg, ToolMessage):
            prompt_messages.append(msg)
        elif isinstance(msg, AIMessage):
            # AIMessage 也直接保留，可能包含 tool_calls
            prompt_messages.append(msg)
        elif isinstance(msg, HumanMessage):
            h = History(role="human", content=msg.content)
            prompt_messages.append(h.to_msg_template(is_raw=True))
        else:
            # 其他类型消息使用 History 转换
            h = History(role=msg.type, content=msg.content)
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
    
    # 调试：打印实际发送给模型的消息
    print("\n=== 发送给模型的消息 ===")
    print(f"历史消息数量: {len(state.messages)}")
    for i, msg in enumerate(state.messages):
        print(f"  [{i}] {msg.type}: {msg.content[:100]}...")
    print(f"\nContext: {final_context[:200]}...")
    print(f"Question: {state.query if state.query else state.messages[-1].content}")
    print("=" * 50 + "\n")
    
    response = await chain.ainvoke({
        "context": final_context,
        "question": state.query if state.query else state.messages[-1].content,
    }, config)
    
    return {"messages": [response]}