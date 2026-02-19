from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState
from core.state_graph.tool.tools import execute_sql_query

def execute_sql(state: SQLQueryState) -> dict:
    """执行SQL查询"""
    result = execute_sql_query.invoke({"sql": state.sql})
    return {"context": result}