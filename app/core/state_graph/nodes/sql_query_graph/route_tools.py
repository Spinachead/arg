from core.state_graph.tool.tools import GENERAL_TOOLS
from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState


def route_tools(state: SQLQueryState):
    """判断是否需要调用工具，并路由到对应的工具节点"""
    messages = state.messages
    last_message = messages[-1]
    
    # 检查是否有工具调用
    if not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
        print("没有调用工具")
        return "end"
    
    # 获取工具名称并路由到对应的节点
    tool_name = last_message.tool_calls[0].get('name')
    
    # 动态获取 general_tools 的工具名称列表
    general_tool_names = [tool.name for tool in GENERAL_TOOLS]
    
    if tool_name in general_tool_names:
        print(f"开始调用工具 {tool_name}")
        return "general_tools"