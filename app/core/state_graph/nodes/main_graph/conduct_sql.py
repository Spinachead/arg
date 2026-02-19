from typing import Any
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.sql_query_graph import sql_query_graph
async def conduct_sql(state: AgentState) -> dict[str, Any]:
    """
    执行 knowledge_graph 子图节点
    """
    
    # 调用知识查询子图
    response = await sql_query_graph.ainvoke({
        "messages": state.messages,
    })

    print(f"conduct_sql:{response}")
    # 将子图的 context 返回给主图状态
    return {
        "context": response.get("context", ""),
    }
