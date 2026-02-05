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
    
    # 获取 MCP 工具
    mcp_tools_dict = cl.user_session.get("mcp_tools", {})
    mcp_tools = []
    for connection_name, tools in mcp_tools_dict.items():
        mcp_tools.extend(tools)
    
    # 绑定工具
    if mcp_tools:
        model = model.bind_tools(mcp_tools)
    
    prompt_template = get_prompt_template("rag", "default")
    system_prompt = f"""{prompt_template}
    [环境上下文]
    - 当前用户 ID: {state.user_id}
    - 请根据工具的描述，在必要时调用它们以获取准确信息。
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

    return {"messages": [response]}