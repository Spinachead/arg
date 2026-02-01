from typing import List, Optional
from db.models.message_model import MessageModel
from db.session import with_session
import uuid

@with_session
def add_message_to_db(
    session,
    thread_id: str,
    query: str,
    response: str,
    chat_type: str = "kb_chat",
    message_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    trace_id: Optional[str] = None
):
    """
    保存对话消息到数据库
    """
    if not message_id:
        message_id = str(uuid.uuid4())
    
    message = MessageModel(
        id=message_id,
        threadId=thread_id,
        chatType=chat_type,
        query=query,
        response=response,
        metadata_=metadata or {},
        traceId=trace_id
    )
    session.add(message)
    return message_id

@with_session
def get_messages_by_thread_id(session, thread_id: str) -> List[MessageModel]:
    """
    根据 threadId 查询对话历史
    """
    messages = (
        session.query(MessageModel)
        .filter(MessageModel.threadId == thread_id)
        .order_by(MessageModel.createdAt.asc())
        .all()
    )
    return messages

@with_session
def delete_messages_by_thread_id(session, thread_id: str):
    """
    删除指定 threadId 的所有对话消息
    """
    session.query(MessageModel).filter(MessageModel.threadId == thread_id).delete()
    return True

@with_session
def get_message_by_id(session, message_id: str) -> Optional[MessageModel]:
    """
    根据消息 ID 获取单条消息
    """
    return session.query(MessageModel).filter(MessageModel.id == message_id).first()
