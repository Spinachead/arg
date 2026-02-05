import json
import chainlit as cl
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from core.state_graph.states.main_graph.agent_state import AgentState
from langchain_core.messages import ToolMessage


async def mcp_tool_node(state: AgentState, config: RunnableConfig) -> Dict[str, Any]:
    """处理 MCP 工具调用"""
    messages = state.messages
    last_message = messages[-1]
    
    # 获取工具调用
    tool_calls = getattr(last_message, 'tool_calls', [])
    if not tool_calls:
        return {"messages": []}
    
    # 获取 MCP 工具字典
    mcp_tools_dict = cl.user_session.get("mcp_tools", {})
    
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call.get('name')
        tool_input = tool_call.get('args', {})
        tool_call_id = tool_call.get('id')
        
        # 创建 Chainlit step 显示工具调用 - 修复 name 字段
        async with cl.Step(type="tool", name=f"🔧 {tool_name}") as step:
            step.input = json.dumps(tool_input, ensure_ascii=False, indent=2)
            
            # 找到对应的 MCP 连接
            mcp_name = None
            for connection_name, tools in mcp_tools_dict.items():
                if any(tool.get("name") == tool_name for tool in tools):
                    mcp_name = connection_name
                    break
            
            if not mcp_name:
                result = json.dumps({"error": f"Tool {tool_name} not found"}, ensure_ascii=False)
            else:
                mcp_session, _ = cl.context.session.mcp_sessions.get(mcp_name)
                if not mcp_session:
                    result = json.dumps({"error": f"MCP {mcp_name} not connected"}, ensure_ascii=False)
                else:
                    try:
                        # 调用远程 MCP 工具
                        mcp_result = await mcp_session.call_tool(tool_name, tool_input)
                        result = str(mcp_result)
                        print(f"[mcp_tool_node] 工具 {tool_name} 执行成功")
                    except Exception as e:
                        result = json.dumps({"error": str(e)}, ensure_ascii=False)
                        print(f"[mcp_tool_node] 工具 {tool_name} 执行失败: {e}")
            
            step.output = result
        
        # 创建工具消息
        tool_message = ToolMessage(
            content=result,
            tool_call_id=tool_call_id,
            name=tool_name
        )
        tool_messages.append(tool_message)
    
    print(f"[mcp_tool_node] 返回 {len(tool_messages)} 条工具消息")
    return {"messages": tool_messages}
