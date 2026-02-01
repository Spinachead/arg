from sqlalchemy import Column, String, JSON, ForeignKey
from db.base import Base

class ThreadModel(Base):
    """
    对话线程模型
    """
    __tablename__ = "threads"

    id = Column(String(36), primary_key=True, comment="线程ID")
    createdAt = Column(String(50), name="createdAt", comment="创建时间")
    name = Column(String(255), comment="线程名称")
    userId = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), name="userId", comment="用户ID")
    userIdentifier = Column(String(255), name="userIdentifier", comment="用户标识符")
    tags = Column(JSON, comment="标签")
    metadata = Column(JSON, name="metadata", comment="元数据")

    def __repr__(self):
        return f"<Thread(id='{self.id}', name='{self.name}', userId='{self.userId}', createdAt='{self.createdAt}')>"
