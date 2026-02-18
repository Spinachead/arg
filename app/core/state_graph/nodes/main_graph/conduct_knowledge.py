from typing import Any
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.knowledge_query_graph import knowledge_graph


async def conduct_knowledge(state: AgentState) -> dict[str, Any]:
    """
    执行konledge_graph节点
    """

    response = await knowledge_graph.ainvoke(
        
    )
    
