from token import STAR
from langgraph.graph import END, START, StateGraph
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.nodes.main_graph.generate_queries import generate_queries
from core.state_graph.nodes.main_graph.retrieve_documents import retrieve_documents
from core.state_graph.nodes.main_graph.respond import respond
from core.state_graph.nodes.main_graph.mcp_tool_node import mcp_tool_node
from langgraph.prebuilt import ToolNode
from core.state_graph.nodes.main_graph.tools import GENERAL_TOOLS
from langgraph.checkpoint.memory import InMemorySaver

# 全局共享的 checkpoint，确保对话历史不会丢失
_GLOBAL_CHECKPOINT = InMemorySaver()


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


def build_main_graph():
    # 使用全局共享的 checkpoint，确保对话历史持久化
    builder = StateGraph(AgentState, input=InputState)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("response", respond)
    builder.add_node("mcp_tools", mcp_tool_node)
    builder.add_node("general_tools", ToolNode(tools=GENERAL_TOOLS))
    
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "retrieve_documents")
    builder.add_edge("retrieve_documents", "response")
    
    # 添加条件边：response -> mcp_tools / general_tools / END
    builder.add_conditional_edges(
        "response",
        route_tools,
        {
            "mcp_tools": "mcp_tools",
            "general_tools": "general_tools",
            "end": END
        }
    )
    
    # 两个工具执行后都回到 response
    builder.add_edge("mcp_tools", "response")
    builder.add_edge("general_tools", "response")
    return builder.compile(checkpointer=_GLOBAL_CHECKPOINT)
