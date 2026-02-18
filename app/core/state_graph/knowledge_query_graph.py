from langgraph.graph import END, START, StateGraph
from core.state_graph.states.knowledge_query_graph.researcher_state import ResearcherState
from core.state_graph.nodes.knowledge_query_graph.generate_queries import generate_queries
from core.state_graph.nodes.knowledge_query_graph.retrieve_documents import retrieve_documents
from core.state_graph.nodes.knowledge_query_graph.respond import respond


def build_knowledge_graph():
    builder = StateGraph(ResearcherState)
    builder.add_node(generate_queries)
    builder.add_node(retrieve_documents)
    builder.add_node(respond)

    builder.add_edge(START, "generate_queries")
    builder.add_edge("generate_queries", "retrieve_documents")
    builder.add_edge("retrieve_documents", END)
    return builder.compile()

knowledge_graph = build_knowledge_graph()
