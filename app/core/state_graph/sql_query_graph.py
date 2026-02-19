from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState
from langgraph.graph import END, START, StateGraph
from core.state_graph.nodes.sql_query_graph.generate_sql import generate_sql
from core.state_graph.nodes.sql_query_graph.generate_answer import generate_answer
from core.state_graph.tool.tools import GENERAL_TOOLS
from langgraph.prebuilt import ToolNode
from core.state_graph.nodes.sql_query_graph.route_tools import route_tools


def build_sql_query_graph():
    builder = StateGraph(SQLQueryState)
    builder.add_node(generate_sql)
    builder.add_node(generate_answer)
    builder.add_node("general_tools", ToolNode(tools=GENERAL_TOOLS))

    builder.add_edge(START, "generate_sql")
    builder.add_edge("generate_sql", "generate_answer")
    builder.add_conditional_edges(
        "generate_answer",
        route_tools,
        {
            "general_tools": "general_tools",
            "end": END
        }
    )
    builder.add_edge("general_tools", "generate_answer")
    return builder.compile()

sql_query_graph = build_sql_query_graph()

sql_query_graph = build_sql_query_graph()

