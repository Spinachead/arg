from token import STAR
from langgraph.graph import END, START, StateGraph
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.states.main_graph.input_state import InputState
from core.state_graph.nodes.main_graph.generate_queries import generate_queries
from core.state_graph.nodes.main_graph.retrieve_documents import retrieve_documents
from core.state_graph.nodes.main_graph.respond import respond



def build_main_graph():
    builder = StateGraph(AgentState, input=InputState)
    builder.add_node("generate_queries", generate_queries)
    builder.add_node("retrieve_documents", retrieve_documents)
    builder.add_node("response", respond)
    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "retrieve_documents")
    builder.add_edge("retrieve_documents", "response")
    builder.add_edge("response", END)
    return builder.compile()
