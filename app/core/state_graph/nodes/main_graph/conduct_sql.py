# 修改 conduct_sql.py
from typing import Any
from core.state_graph.states.main_graph.agent_state import AgentState
from core.state_graph.sql_query_graph import sql_query_graph
from langchain_core.messages import AIMessage, HumanMessage

async def conduct_sql(state: AgentState) -> dict[str, Any]:
    """
    执行 SQL 查询子图节点
    """
    # 提取最后一条用户消息作为当前问题
    current_question = None
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            current_question = msg
            break
    
    if not current_question:
        return {"messages": state.messages, "context": ""}

    # 调用 SQL 查询子图，只传递当前问题
    response = await sql_query_graph.ainvoke({
        "messages": [current_question],  # 只传递当前问题，不包含历史
    })

    print(f"\033[92mUsing conduct_sql: {response}\033[0m")  # 绿色输出
    

    # 从子图响应中提取最后一条 AI 消息作为回答
    messages = response.get("messages", [])
    sql_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            sql_answer = msg.content
            break

    # 将子图的回答和上下文返回给主图状态
    messages = state.messages + response.get("messages", [])
    return {
        "messages": messages,
        "context": response.get("context", ""),
    }