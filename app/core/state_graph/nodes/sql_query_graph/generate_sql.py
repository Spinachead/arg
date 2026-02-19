from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from settings import Settings
from langchain.chat_models import init_chat_model
from core.state_graph.states.sql_query_graph.sql_query import SQLQueryState
from core.prompts import GENERATE_SQL_PROMPT
from db.db_schema import DB_SCHEMA

async def generate_sql(state: SQLQueryState, *, config: RunnableConfig) -> dict:
    """
    生成sql查询语句
    """
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
    print(f"generate_sql: {response}")
    return {"sql": response}