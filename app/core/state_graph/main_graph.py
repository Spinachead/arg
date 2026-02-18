from langgraph.graph import END, START, StateGraph
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.nodes.main_graph.respond import respond
from core.state_graph.nodes.main_graph.mcp_tool_node import mcp_tool_node
from langgraph.prebuilt import ToolNode
from core.state_graph.nodes.main_graph.tools import GENERAL_TOOLS
from langgraph.checkpoint.memory import InMemorySaver
from core.state_graph.nodes.main_graph.analyze_and_route_query import analyze_and_route_query
from core.state_graph.nodes.main_graph.ask_for_more_info import ask_for_more_info
from core.state_graph.nodes.main_graph.respond_to_general_query import respond_to_general_query
from core.state_graph.nodes.main_graph.conduct_knowledge import conduct_knowledge
from core.state_graph.nodes.main_graph.route_query import route_query
from core.state_graph.nodes.main_graph.route_tools import route_tools
# 全局共享的 checkpoint，确保对话历史不会丢失
_GLOBAL_CHECKPOINT = InMemorySaver()

def build_main_graph2():
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


def build_main_graph():
    builder = StateGraph(AgentState, input=InputState)
    builder.add_node(analyze_and_route_query)
    builder.add_node(ask_for_more_info)
    builder.add_node(respond_to_general_query)
    builder.add_node(conduct_knowledge)
    builder.add_node("respond", respond)

    builder.add_edge(START, "analyze_and_route_query")
    builder.add_conditional_edges("analyze_and_route_query", route_query)
    builder.add_edge("conduct_knowledge", "respond")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=_GLOBAL_CHECKPOINT)