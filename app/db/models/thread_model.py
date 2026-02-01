from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from db.base import Base

class ThreadModel(Base):
    """
    对话线程模型
    """
    __tablename__ = "threads"

    id = Column(UUID(as_uuid=False), primary_key=True, comment="线程ID")
    createdAt = Column(String(50), name="createdAt", comment="创建时间")
    name = Column(String(255), comment="线程名称")
    userId = Column(UUID(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), name="userId", comment="用户ID")
    userIdentifier = Column(String(255), name="userIdentifier", comment="用户标识符")
    tags = Column(JSONB, comment="标签")
    metadata_ = Column(JSONB, name="metadata", comment="元数据")

    def __repr__(self):
        return f"<Thread(id='{self.id}', name='{self.name}', userId='{self.userId}', createdAt='{self.createdAt}')>"
