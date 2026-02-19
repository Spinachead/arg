from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState
from langgraph.graph import END, START, StateGraph
from core.state_graph.nodes.sql_query_graph.generate_sql import generate_sql
from core.state_graph.nodes.sql_query_graph.generate_answer import generate_answer
from core.state_graph.nodes.sql_query_graph.execute_sql import execute_sql

def build_sql_query_graph():
    builder = StateGraph(SQLQueryState)
    builder.add_node(generate_sql)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node(generate_answer)

    builder.add_edge(START, "generate_sql")
    builder.add_edge("generate_sql", "execute_sql")
    builder.add_edge("execute_sql", "generate_answer")
    builder.add_edge("generate_answer", END)
    return builder.compile()

sql_query_graph = build_sql_query_graph()

