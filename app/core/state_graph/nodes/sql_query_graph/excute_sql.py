from core.state_graph.states.research_graph.query_state import QueryState
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from settings import Settings
from langchain.chat_models import init_chat_model
from utils import History, build_logger
from tools import execute_sql_query

def execute_sql(state: QueryState) -> dict:
    """
    执行sql并且返回查询结果
    """

    try:
        result = execute_sql_query.invoke(state.query)
        return {"sql_result": result}
    except Exception as e:
        return {"sql_result": f"查询出错：{str(e)}"}
