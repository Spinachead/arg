from settings import Settings
from langchain_core.prompts import ChatPromptTemplate
from utils import History, build_logger
from langchain.chat_models import init_chat_model
from db.repository.knowledge_base_repository import list_kbs_from_db
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
from core.state_graph.states.knowledge_query_graph.researcher_state import ResearcherState
from core.state_graph.states.knowledge_query_graph.multi_query_result import MultiQueryResult
from core.state_graph.states.knowledge_query_graph.query_state import QueryState
logger = build_logger()

async def generate_queries(state:QueryState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    生成多个查询变体以及其对应的知识库名称
    """
    model = init_chat_model(
        name="generate_queries",
        model=Settings.app_settings.inference_model,
        temperature=Settings.app_settings.temperature,
        streaming=Settings.app_settings.streaming,
        openai_api_base=Settings.app_settings.openai_api_base,
        openai_api_key=Settings.app_settings.openai_api_key,
    )
    structured_llm = model.with_structured_output(MultiQueryResult)

    query_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的查询分析和改写助手。根据用户的原始查询，你需要：
        1. 生成3个不同角度或表述的查询变体，用于从知识库中检索相关信息
        """),
        ("human", "原始查询: {query}")
    ])

    try:
        result = await structured_llm.ainvoke(
            query_gen_prompt.format(
                query=state.query, 
            ),
            config,
        )
        return {"queries": result}
    except Exception as e:
        logger.warning(f"结构化输出失败，使用备用方案: {e}")
        return {"queries": [state.query]}








