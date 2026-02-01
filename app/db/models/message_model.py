from sqlalchemy import JSON, Column, DateTime, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID

from db.base import Base


class MessageModel(Base):
    """
    聊天记录模型
    """

    __tablename__ = "message"
    id = Column(UUID(as_uuid=False), primary_key=True, comment="聊天记录ID")
    threadId = Column(UUID(as_uuid=False), name="threadId", index=True, comment="对话框ID")
    chatType = Column(String(50), name="chatType", comment="聊天类型")
    query = Column(String(4096), comment="用户问题")
    response = Column(String(4096), comment="模型回答")
    # 记录知识库id等，以便后续扩展
    metadata_ = Column(JSON, name="metadata", default={})
    # LangSmith trace id：langsmith.trace.get_current_run_tree().id
    traceId = Column(String(255), name="traceId", index=True, comment="LangSmith Trace Run ID")
    createdAt = Column(String(50), name="createdAt", default=func.now(), comment="创建时间")

    def __repr__(self):
        return f"<message(id='{self.id}', threadId='{self.threadId}', chatType='{self.chatType}', query='{self.query}', response='{self.response}', metadata='{self.metadata_}', createdAt='{self.createdAt}')>"
