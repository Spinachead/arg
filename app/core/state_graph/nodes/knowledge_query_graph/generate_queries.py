from settings import Settings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from utils import History, build_logger
from langchain.chat_models import init_chat_model
from db.repository.knowledge_base_repository import list_kbs_from_db
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any, TypedDict
from core.state_graph.states.knowledge_query_graph.researcher_state import ResearcherState
from core.prompts import GENERATE_QUERIES
logger = build_logger()

async def generate_queries(state:ResearcherState, *, config: RunnableConfig) -> Dict[str, Any]:
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
    class Response(TypedDict):
        queries: list[str]
    structured_llm = model.with_structured_output(Response)
    query_gen_prompt = ChatPromptTemplate.from_messages([
       ("system", GENERATE_QUERIES),
       MessagesPlaceholder(variable_name="history"),
    ])

    try:
        messages = query_gen_prompt.format_messages(history=state.messages)
        result = await structured_llm.ainvoke(
            messages,
            config,
        )
        print(f"generate_queries:{result}")
        return {"queries": result.get("queries")}
    except Exception as e:
        logger.warning(f"结构化输出失败，使用备用方案: {e}")
        return {"queries": [""]}








