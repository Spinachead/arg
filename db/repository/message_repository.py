from db.models.message_model import MessageModel
from db.session import with_session


@with_session
def add_message_to_db(
    session,
    message_id: str,
    conversation_id: str,
    chat_type: str,
    query: str,
    response: str,
    trace_id: str = None,
    meta_data: dict = None,
):
    """
    添加消息记录到数据库
    """
    message = MessageModel(
        id=message_id,
        conversation_id=conversation_id,
        chat_type=chat_type,
        query=query,
        response=response,
        trace_id=trace_id,
        meta_data=meta_data or {},
    )
    session.add(message)
    return message_id


@with_session
def get_message_by_id(session, message_id: str):
    """
    根据消息ID获取消息
    """
    message = (
        session.query(MessageModel)
        .filter(MessageModel.id == message_id)
        .first()
    )
    return message


@with_session
def update_message_feedback(
    session,
    message_id: str,
    feedback_score: int,
    feedback_reason: str = "",
):
    """
    更新消息的评价反馈
    """
    message = (
        session.query(MessageModel)
        .filter(MessageModel.id == message_id)
        .first()
    )
    if message:
        message.feedback_score = feedback_score
        message.feedback_reason = feedback_reason
        return True
    return False


@with_session
def list_messages_by_conversation(session, conversation_id: str, limit: int = 50):
    """
    根据对话ID获取消息列表
    """
    messages = (
        session.query(MessageModel)
        .filter(MessageModel.conversation_id == conversation_id)
        .order_by(MessageModel.create_time.desc())
        .limit(limit)
        .all()
    )
    return messages
