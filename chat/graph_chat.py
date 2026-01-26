import json
import os
from typing import TypedDict, List, Dict, Any, Annotated
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END, add_messages
from pydantic import BaseModel, Field
from langchain_core.tools import tool

from db.repository.user_memory_repository import get_user_profile_from_memories
from db.repository.knowledge_base_repository import list_kbs_from_db
from knowledge_base.kb_doc_api import search_docs
from utils import History, build_logger, get_ChatOpenAI, get_config_models, format_reference,get_prompt_template
from api_schemas import MultiQueryResult
from langgraph.checkpoint.memory import MemorySaver

logger = build_logger()
load_dotenv()

class GraphChatState(TypedDict):
    query: str
    kb_name: str
    top_k: int
    score_threshold: float
    model: str
    temperature: float
    user_id: int
    query_kb_pairs: List[Dict[str, str]]
    context: str
    sources: List[str]
    messages: Annotated[list[AnyMessage], add_messages]

 # 获取模型配置
model_info = get_config_models(model_name="qwen-max", model_type="llm")
model_config = model_info["qwen-max"]
llm = get_ChatOpenAI(
    model_name="qwen-max",
    openai_api_base=model_config["api_base_url"],
    openai_api_key=model_config["api_key"],
)


@tool
def load_user_profile_tool(user_id: int) -> Dict[str, Any]:
    """
    获取用户相关的记忆，如喜好、偏好、最近话题、个人信息等。
    当需要了解用户的偏好、习惯或历史信息时调用此工具。
    
    Args:
        user_id: 用户ID
    
    Returns:
        包含用户画像信息的字典，包括preferred_kbs(偏好知识库)、preferred_domains(偏好领域)、tone(语气)、recent_topics(最近话题)等
    """
    user_profile = {"preferred_kbs": [], "preferred_domains": [], "tone": "标准", "recent_topics": []}
    try:
        user_profile = get_user_profile_from_memories(user_id=user_id)
        logger.info(f"已加载用户 {user_id} 的长期记忆: {user_profile}")
    except Exception as e:
        logger.warning(f"获取用户记忆失败: {e}")
    return user_profile


tools = [load_user_profile_tool]
llm_with_tools = llm.bind_tools(tools)


def generate_queries_node(state: GraphChatState) -> Dict[str, Any]:
    """生成多个查询变体及其对应的知识库名称"""
    query = state["query"]
    kb_name = state["kb_name"]
    
    available_kbs = list_kbs_from_db()
    kb_info_str = "\n".join([f"- {kb.kb_name}: {kb.kb_info or '无描述'}" for kb in available_kbs])
    
    structured_llm = llm.with_structured_output(MultiQueryResult)
    
    query_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的查询分析和改写助手。根据用户的原始查询，你需要：
        1. 生成3个不同角度或表述的查询变体，用于从知识库中检索相关信息
        2. 为每个查询变体选择最合适的知识库

        可用的知识库列表：{kb_list}

        请返回JSON格式，包含queries数组，每个元素有query（查询文本）和kb_name（知识库名称）两个字段。"""),
        ("human", "原始查询: {query}")
    ])
    
    try:
        result = structured_llm.invoke(
            query_gen_prompt.format(
                query=query, 
                kb_list=kb_info_str,
            )
        )
        query_kb_pairs = [{"query": q.query, "kb_name": q.kb_name} for q in result.queries]
        if len(query_kb_pairs) < 3:
            query_kb_pairs.insert(0, {"query": query, "kb_name": kb_name})
        return {"query_kb_pairs": query_kb_pairs}
    except Exception as e:
        logger.warning(f"结构化输出失败，使用备用方案: {e}")
        return {"query_kb_pairs": [{"query": query, "kb_name": kb_name}]}

def retrieve_documents_node(state: GraphChatState) -> Dict[str, Any]:
    """使用多个查询变体检索文档并合并结果"""
    query_kb_pairs = state.get("query_kb_pairs", [])
    top_k = state.get("top_k", 3)
    score_threshold = state.get("score_threshold", 0.5)
    kb_name = state.get("kb_name", "")
    original_query = state.get("query", "")
    
    all_docs = []
    doc_id_set = set()
    
    for pair in query_kb_pairs:
        q = pair["query"]
        target_kb = pair["kb_name"]
        kb_to_use = target_kb if target_kb else kb_name
        
        docs = search_docs(
            query=q,
            knowledge_base_name=kb_to_use,
            top_k=top_k,
            score_threshold=score_threshold,
            file_name="",
            metadata={}
        )
        for doc in docs:
            doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
            if doc_id and doc_id not in doc_id_set:
                doc_id_set.add(doc_id)
                all_docs.append(doc)
    
    if not all_docs:
        docs = search_docs(
            query=original_query,
            knowledge_base_name=kb_name,
            top_k=top_k,
            score_threshold=score_threshold,
            file_name="",
            metadata={}
        )
        all_docs = docs
    
    source_documents = format_reference(kb_name, all_docs, "")
    context = "\n\n".join([doc.get("page_content", "") for doc in all_docs])
    
    return {
        "context": context,
        "sources": source_documents,
    }

def llm_call_node(state: GraphChatState) -> Dict[str, Any]:
    """ 调用LLM生成最终回复 """
    prompt_template = get_prompt_template("rag", "default")
    from langchain_core.prompts import ChatPromptTemplate
    chat_prompt = ChatPromptTemplate.from_messages([
        History(role="user", content=prompt_template).to_msg_template(False)
    ])

    response = llm_with_tools.invoke(
        chat_prompt.format(
                context=state["context"],
                sources=state["sources"] if state["sources"] else "未知来源",
                question=state["query"],
        )
    )

    return {"messages": [response]}

def create_graph_chat_app():
    workflow = StateGraph(GraphChatState)
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("llm_call", llm_call_node)
    
    workflow.add_edge(START, "generate_queries")
    workflow.add_edge("generate_queries", "retrieve")
    workflow.add_edge("retrieve", "llm_call")
    workflow.add_edge("llm_call", END)
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)
