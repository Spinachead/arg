from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from settings import Settings
from langchain.chat_models import init_chat_model
from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState
from core.prompts import GENERATE_SQL_PROMPT
from db.db_schema import DB_SCHEMA

async def generate_sql(state: SQLQueryState, *, config: RunnableConfig) -> dict:
    """
    生成sql查询语句，并调用工具执行
    """
    # 绑定工具，让模型可以调用 execute_sql_query
    model = init_chat_model(
        name="generate_sql",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATE_SQL_PROMPT),
        MessagesPlaceholder("history"),
    ])
    chain = prompt | model
    response = await chain.ainvoke({
        "history": state.messages,
        "DB_SCHEMA": DB_SCHEMA
        }, config)
    print(f"\033[92mUsing generate_sql: {response}\033[0m")  # 绿色输出
    
    # 提取纯净的 SQL，去除 Markdown 代码块标记
    sql = response.content.strip()
    if sql.startswith("```sql"):
        sql = sql[6:].strip()
    elif sql.startswith("```"):
        sql = sql[3:].strip()
    if sql.endswith("```"):
        sql = sql[:-3].strip()
    return {"sql": sql}
