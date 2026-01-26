import json
import os
from typing import TypedDict, List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
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
    user_profile: Dict[str, Any]
    query_kb_pairs: List[Dict[str, str]]
    context: str
    sources: List[str]
    answer: str  # 新增：存储LLM的最终回答


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

def decide_load_profile_node(state: GraphChatState) -> Dict[str, Any]:
    """
    让LLM决定是否需要加载用户画像
    如果查询涉及个性化推荐或需要了解用户偏好时，才调用工具
    """
    query = state["query"]
    user_id = state.get("user_id", 1)
    
    # 让LLM判断是否需要用户画像
    decision_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个智能助手。分析用户的查询，判断是否需要加载用户的个人画像。
        
如果查询涉及以下情况，应该调用load_user_profile_tool：
- 个性化推荐（例如："给我推荐..."、"我想了解..."）
- 需要了解用户偏好（例如："我常用的..."、"我喜欢的..."）
- 上下文相关的历史话题

如果是一般性的事实查询，不需要调用工具。

可用工具：load_user_profile_tool(user_id)"""),
        ("human", "用户查询: {query}\n\n用户ID: {user_id}\n\n请决定是否需要调用load_user_profile_tool。")
    ])
    
    # 调用带工具的LLM
    response = llm_with_tools.invoke(
        decision_prompt.format_messages(query=query, user_id=user_id)
    )
    
    # 检查是否有工具调用
    user_profile = state.get("user_profile", {})
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tool_call in response.tool_calls:
            if tool_call['name'] == 'load_user_profile_tool':
                # 执行工具调用
                user_profile = load_user_profile_tool.invoke(tool_call['args'])
                logger.info(f"通过tool加载了用户画像: {user_profile}")
                break
    
    return {"user_profile": user_profile}


def generate_queries_node(state: GraphChatState) -> Dict[str, Any]:
    """生成多个查询变体及其对应的知识库名称"""
    query = state["query"]
    user_profile = state["user_profile"]
    kb_name = state["kb_name"]
    
    # 获取所有可用的知识库列表
    available_kbs = list_kbs_from_db()
    kb_info_str = "\n".join([f"- {kb.kb_name}: {kb.kb_info or '无描述'}" for kb in available_kbs])
    
    # 使用结构化输出（这里不需要工具绑定，因为是结构化输出）
    structured_llm = llm.with_structured_output(MultiQueryResult)
    
    # 生成查询变体和知识库匹配的提示
    query_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", """你是一个专业的查询分析和改写助手。根据用户的原始查询，你需要：
        1. 生成3个不同角度或表述的查询变体，用于从知识库中检索相关信息
        2. 为每个查询变体选择最合适的知识库

        用户的长期偏好信息：
        {user_profile}
        
        在选择知识库时，请优先考虑用户常用或偏好的知识库（preferred_kbs）和领域（preferred_domains）。

        可用的知识库列表：{kb_list}

        请返回JSON格式，包含queries数组，每个元素有query（查询文本）和kb_name（知识库名称）两个字段。"""),
        ("human", "原始查询: {query}")
    ])
    
    try:
        result = structured_llm.invoke(
            query_gen_prompt.format(
                query=query, 
                kb_list=kb_info_str,
                user_profile=json.dumps(user_profile, ensure_ascii=False)
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
    
    # 提取LLM回复的文本内容
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # 返回状态更新（必须是GraphChatState的字段）
    return {"answer": answer}
   


def create_graph_chat_app():
    workflow = StateGraph(GraphChatState)
    workflow.add_node("decide_profile", decide_load_profile_node)  # 新增：决定是否加载用户画像
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("retrieve", retrieve_documents_node)
    workflow.add_node("llm_call", llm_call_node)
    
    workflow.add_edge(START, "decide_profile")  # 从开始先决定是否加载用户画像
    workflow.add_edge("decide_profile", "generate_queries")  # 然后生成查询变体
    workflow.add_edge("generate_queries", "retrieve")
    workflow.add_edge("retrieve", "llm_call")
    workflow.add_edge("llm_call", END)
    checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)
