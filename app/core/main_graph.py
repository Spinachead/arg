from token import STAR
from langgraph.graph import END, START, StateGraph
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.nodes.main_graph.generate_queries import generate_queries
from core.state_graph.nodes.main_graph.retrieve_documents import retrieve_documents
from core.state_graph.nodes.main_graph.respond import respond
from core.state_graph.nodes.main_graph.mcp_tool_node import mcp_tool_node


def should_continue(state: AgentState):
    """判断是否需要继续调用工具"""
    messages = state.messages
    last_message = messages[-1]
    
    # 检查是否有工具调用
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


def build_main_graph():
    builder = StateGraph(AgentState, input=InputState)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("response", respond)
    builder.add_node("tools", mcp_tool_node)
    
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "retrieve_documents")
    builder.add_edge("retrieve_documents", "response")
    
    # 添加条件边：response -> tools 或 END
    builder.add_conditional_edges(
        "response",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    
    # 工具执行后回到 response
    builder.add_edge("tools", "response")
    
    return builder.compile()
