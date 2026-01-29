from core.state_graph.states.main_graph.agent_state import AgentState
from knowledge_base.kb_doc_api import search_docs
from utils import format_reference
from langchain_core.runnables import RunnableConfig
from typing import Dict, Any

async def retrieve_documents(state: AgentState, *, config: RunnableConfig) -> Dict[str, Any]:
    """使用多个查询变体检索文档并合并结果"""
    query_kb_pairs = state.query_kb_pairs
    all_docs = []
    doc_id_set = set()
    
    for pair in query_kb_pairs:
        q = pair["query"]
        target_kb = pair["kb_name"]
        kb_to_use = target_kb if target_kb else "low"
        print(f"正在检索知识库: {kb_to_use}, 查询词: {q}")
        
        docs = search_docs(
            query=q,
            knowledge_base_name=kb_to_use,
            top_k=5,
            score_threshold=2.0,
            file_name="",
            metadata={}
        )
        print(f"检索到的文档：{docs}")
        for doc in docs:
            # 优先从顶级 id 获取，其次从 metadata 获取，最后使用内容哈希防止过滤掉无 ID 的文档
            doc_id = doc.get("id") or doc.get("metadata", {}).get("id")
            if not doc_id:
                import hashlib
                content = doc.get("page_content", "")
                doc_id = hashlib.md5(content.encode()).hexdigest()

            if doc_id not in doc_id_set:
                doc_id_set.add(doc_id)
                all_docs.append(doc)
    
    print(f"检索到的文档内容：{all_docs}")
    source_documents = format_reference("low", all_docs, "")
    context = "\n\n".join([doc.get("page_content", "") for doc in all_docs])
    print(f"这是检索到的文档：\n{context}")
    return {
        "context": context,
        "sources": source_documents,
    }

