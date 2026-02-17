from core.state_graph.states.research_graph.query_state import QueryState
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from settings import Settings
from langchain.chat_models import init_chat_model
from utils import History, build_logger

logger = build_logger()



async def generate_sql(state: QueryState, *, config: RunnableConfig) -> dict:
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
        ("system", f"你是一个MySQL专家。根据用户原始查询和以下数据库schema生成SQL查询语句：\n{DB_SCHEMA}\n只输出合法的SELECT SQL，不要解释。"),
        ("human", "原始查询: {query}")

    ])

    chain = prompt | model

    query = state.messages[-1].content
    try:
        result = await chain.ainvoke(
            prompt.format(
                query=query, 
            ),
            config,
        )
       
        return {"sql": result, "query": query}
    except Exception as e:
        logger.warning(f"生成SQL失败: {e}")
        return {"sql": "", "query": query}


