from core.state_graph.states.main_graph.agent_state import AgentState
from config import config as app_config
from core.state_graph.states.main_graph.multi_query_result import MultiQueryResult
from langchain_core.prompts import ChatPromptTemplate
from utils import History, build_logger
from langchain.chat_models import init_chat_model
from db.repository.knowledge_base_repository import list_kbs_from_db
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

logger = build_logger()


async def generate_queries(state:AgentState, *, config: RunnableConfig) -> Dict[str, Any]:
    """
    生成多个查询变体以及其对应的知识库名称

    """
    model = init_chat_model(name="generate_queries", **app_config["inference_model_params"])
    structured_llm = model.with_structured_output(MultiQueryResult)

    query_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的查询分析和改写助手。根据用户的原始查询，你需要：
        1. 生成3个不同角度或表述的查询变体，用于从知识库中检索相关信息
        2. 为每个查询变体选择最合适的知识库

        可用的知识库列表：{kb_list}

        请返回JSON格式，包含queries数组，每个元素有query（查询文本）和kb_name（知识库名称）两个字段。"""),
        ("human", "原始查询: {query}")
    ])

    query = state.messages[-1].content
    available_kbs = list_kbs_from_db()
    kb_info_str = "\n".join([f"- {kb.kbName}: {kb.kbInfo or '无描述'}" for kb in available_kbs])

    kb_name = available_kbs[0].kbName if available_kbs else "low"
    try:
        result = await structured_llm.ainvoke(
            query_gen_prompt.format(
                query=query, 
                kb_list=kb_info_str,
            ),
            config,
        )
        query_kb_pairs = [{"query": q.query, "kb_name": q.kb_name} for q in result.queries]
        if len(query_kb_pairs) < 3:
            query_kb_pairs.insert(0, {"query": query, "kb_name": kb_name})
        return {"query_kb_pairs": query_kb_pairs, "query": query}
    except Exception as e:
        logger.warning(f"结构化输出失败，使用备用方案: {e}")
        return {"query_kb_pairs": [{"query": query, "kb_name": kb_name}], "query": query}








