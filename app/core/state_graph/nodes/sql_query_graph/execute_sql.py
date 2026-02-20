from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState
from core.state_graph.tool.tools import execute_sql_query

def execute_sql(state: SQLQueryState) -> dict:
    """执行SQL查询"""
    result = execute_sql_query.invoke({"sql": state.sql})
    print(f"\033[92mUsing execute_sql: {result}\033[0m")  # 绿色输出
    return {"context": result}