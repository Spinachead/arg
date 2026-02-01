# db/models/user_memory_model.py
from sqlalchemy import Column, Integer, String, DateTime, func, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from db.base import Base

class UserMemoryModel(Base):
    __tablename__ = "user_memory"

    id = Column(UUID(as_uuid=False), primary_key=True, comment="记忆ID")
    userId = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), name="userId", index=True, comment="用户ID")
    memoryText = Column(String(1024), name="memoryText", comment="记忆内容摘要")
    importance = Column(Integer, default=1, comment="重要程度 1-5")
    lastUsedTime = Column(String(50), name="lastUsedTime", default=func.now(), comment="最近使用时间")
    createdAt = Column(String(50), name="createdAt", default=func.now(), comment="创建时间")
    metadata_ = Column(JSONB, name="metadata", default={})
    threadId = Column(UUID(as_uuid=False), ForeignKey("threads.id", ondelete="SET NULL"), name="threadId", index=True, comment="线程ID")

    def __repr__(self):
        return f"<UserMemory(id='{self.id}', userId='{self.userId}', memoryText='{self.memoryText[:20]}...', importance={self.importance})>"