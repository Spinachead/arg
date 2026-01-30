import chainlit as cl
from core.main_graph import build_main_graph
from core.state_graph.states.main_graph.input_state import InputState
from db.repository.message_repository import list_messages_by_conversation
from langchain_core.messages import HumanMessage, AIMessage


async def execute():
    cl.user_session.set("graph", build_main_graph())
    
    # 获取对话 ID (优先使用用户名实现跨 Session 持久化，否则使用 session ID)
    user = cl.user_session.get("user")
    conversation_id = user.identifier if user else cl.user_session.get("id")

    
    # 从数据库加载历史记录
    history_messages = []
    db_messages = list_messages_by_conversation(conversation_id)
    
    # 按时间正序排列并转换为 LangChain 消息格式
    for msg in reversed(db_messages):
        history_messages.append(HumanMessage(content=msg.query))
        history_messages.append(AIMessage(content=msg.response))
        
        # 在页面初始化时展示历史记录
        await cl.Message(content=msg.query, author="User").send()
        await cl.Message(content=msg.response, author="Assistant").send()


    cl.user_session.set("state", InputState(messages=history_messages))

