from sqlalchemy import Column, String, Integer, ForeignKey
from db.base import Base

class FeedbackModel(Base):
    """
    反馈模型 (Chainlit Feedbacks)
    """
    __tablename__ = "feedbacks"

    id = Column(String(36), primary_key=True, comment="反馈ID")
    forId = Column(String(36), name="forId", nullable=False, comment="关联ID (Step ID)")
    threadId = Column(String(36), ForeignKey("threads.id", ondelete="CASCADE"), name="threadId", nullable=False, comment="线程ID")
    value = Column(Integer, nullable=False, comment="分值")
    comment = Column(String, comment="评论")

    def __repr__(self):
        return f"<Feedback(id='{self.id}', value='{self.value}', threadId='{self.threadId}')>"
