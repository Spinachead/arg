from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from db.base import Base

class FeedbackModel(Base):
    """
    反馈模型 (Chainlit Feedbacks)
    """
    __tablename__ = "feedbacks"

    id = Column(UUID(as_uuid=False), primary_key=True, comment="反馈ID")
    forId = Column(UUID(as_uuid=False), name="forId", nullable=False, comment="关联ID (Step ID)")
    threadId = Column(UUID(as_uuid=False), ForeignKey("threads.id", ondelete="CASCADE"), name="threadId", nullable=False, comment="线程ID")
    value = Column(Integer, nullable=False, comment="分值")
    comment = Column(String, comment="评论")

    def __repr__(self):
        return f"<Feedback(id='{self.id}', value='{self.value}', threadId='{self.threadId}')>"
