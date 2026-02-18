import chainlit as cl
from settings import Settings
from core.state_graph.states.main_graph.agent_state import AgentState
from knowledge_base.kb_doc_api import search_docs
from utils import format_reference
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any
from core.state_graph.states.knowledge_query_graph.researcher_state import ResearcherState

async def retrieve_documents(state: ResearcherState, *, config: RunnableConfig) -> Dict[str, Any]:
    """使用多个查询变体检索文档并合并结果"""
    query_kb_pairs = state.queries
    all_docs = []
    doc_id_set = set()
    
    # 从用户 session 中获取选择的知识库，如果没有则使用默认值
    model_settings = cl.user_session.get("model_settings", {})
    kb_name = model_settings.get("knowledge_base", Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE)
    
    for pair in query_kb_pairs:
        docs = search_docs(
            query=pair,
            knowledge_base_name=kb_name,
            top_k=5,
            score_threshold=2.0,
            file_name="",
            metadata={}
        )
        for doc in docs:
            doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
            if not doc_id:
                import hashlib
                content = doc.get("page_content", "")
                doc_id = hashlib.md5(content.encode()).hexdigest()

            if doc_id not in doc_id_set:
                doc_id_set.add(doc_id)
                all_docs.append(doc)
    
    source_documents = format_reference(kb_name, all_docs, "")
    context = "\n\n".join([doc.get("page_content", "") for doc in all_docs])
    return {
        "context": context,
    }

