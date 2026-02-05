from core.state_graph.states.main_graph.agent_state import AgentState
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage
from config import config as app_config
from utils import get_prompt_template, History
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
import chainlit as cl


async def respond(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """ 调用LLM生成最终回复,支持 MCP 工具调用 """
    model = init_chat_model(name="respond", **app_config["inference_model_params"])
    
    # 检查上一条消息是否是工具消息 (防止无限循环)
    last_message = state.messages[-1] if state.messages else None
    is_after_tool = False
    
    if last_message:
        # 检查是否是 ToolMessage 类型
        from langchain_core.messages import ToolMessage
        is_after_tool = isinstance(last_message, ToolMessage)
    
    print(f"[respond] 当前消息数: {len(state.messages)}, 上一条是工具消息: {is_after_tool}")
    if is_after_tool:
        print(f"[respond] 上一条消息类型: {type(last_message).__name__}")
    
    
    # 获取 MCP 工具
    mcp_tools_dict = cl.user_session.get("mcp_tools", {})
    mcp_tools = []
    for connection_name, tools in mcp_tools_dict.items():
        mcp_tools.extend(tools)
    
    print(f"[respond] 获取到 {len(mcp_tools)} 个 MCP 工具")
    for i, tool in enumerate(mcp_tools):
        print(f"  工具 {i+1}: {tool.get('name')} - {tool.get('description', 'N/A')[:50]}")
    
    # 绑定工具 - 如果是工具执行后的回复,不再绑定工具
    if mcp_tools and not is_after_tool:
        # 临时测试: 如果用户消息包含"测试工具",强制调用工具
        user_question = state.query if state.query else state.messages[-1].content
        
        # 检测是否需要强制调用工具
        force_tool = False
        if "测试工具" in user_question:
            force_tool = True
            print(f"[respond] ⚠️ 检测到'测试工具'关键词,强制工具调用模式")
        elif any(keyword in user_question for keyword in ["创建实体", "搜索节点", "读取图谱", "添加观察"]):
            force_tool = True
            print(f"[respond] ⚠️ 检测到图谱操作关键词,启用强制工具调用")
        
        if force_tool:
            try:
                # 尝试使用 tool_choice="any" 强制模型调用至少一个工具
                model = model.bind_tools(mcp_tools, tool_choice="any")
                print(f"[respond] 使用 tool_choice='any' 强制工具调用")
            except Exception as e:
                print(f"[respond] ⚠️ tool_choice 不支持: {e}, 回退到普通模式")
                model = model.bind_tools(mcp_tools)
        else:
            model = model.bind_tools(mcp_tools)
        
        print(f"[respond] 已绑定工具到模型")
    elif is_after_tool:
        print(f"[respond] 🔒 工具执行后,不再绑定工具,避免无限循环")
    else:
        print(f"[respond] 警告: 没有可用的 MCP 工具!")
    
    prompt_template = get_prompt_template("rag", "default")
    system_prompt = f"""{prompt_template}
    [环境上下文]
    - 当前用户 ID: {state.user_id}
    
    [重要指示]
    你可以使用以下工具来增强回答能力:
    1. 当用户要求"记住"、"保存"某些信息时,必须调用相关的存储工具
    2. 当用户询问"之前说过什么"、"我记了什么"时,必须调用相关的查询工具
    3. 当需要访问外部数据或执行特定操作时,优先考虑使用可用的工具
    4. 如果用户的问题明确需要工具才能完成,请主动调用工具,不要仅凭已知信息推测
    
    请根据工具的描述,在必要时主动调用它们以获取准确信息。
    """
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        History(role="user", content=prompt_template).to_msg_template(False)
    ])

    chain = chat_prompt | model
    response = await chain.ainvoke({
        "context": state.context,
        "sources": state.sources if state.sources else "未知来源",
        "question": state.query if state.query else state.messages[-1].content,
    }, config)

    print(f"[respond] 响应类型: {type(response)}")
    print(f"[respond] 响应内容: {response.content[:100] if hasattr(response, 'content') else 'N/A'}")
    print(f"[respond] 是否有 tool_calls 属性: {hasattr(response, 'tool_calls')}")
    if hasattr(response, 'tool_calls'):
        print(f"[respond] tool_calls 内容: {response.tool_calls}")
    
    return {"messages": [response]}