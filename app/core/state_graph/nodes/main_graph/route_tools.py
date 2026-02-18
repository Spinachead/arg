from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.nodes.main_graph.tools import GENERAL_TOOLS


def route_tools(state: AgentState):
    """判断是否需要调用工具，并路由到对应的工具节点"""
    messages = state.messages
    last_message = messages[-1]
    
    # 检查是否有工具调用
    if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        return "end"
    
    # 获取工具名称并路由到对应的节点
    tool_name = last_message.tool_calls[0].get('name')
    
    # 动态获取 general_tools 的工具名称列表
    general_tool_names = [tool.name for tool in GENERAL_TOOLS]
    
    if tool_name in general_tool_names:
        return "general_tools"
    else:
        # 默认路由到 MCP 工具节点
        return "mcp_tools"