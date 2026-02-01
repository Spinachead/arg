from sqlalchemy import Column, String, JSON
from sqlalchemy.dialects.postgresql import UUID
from db.base import Base

class UserModel(Base):
    """
    用户模型
    """
    __tablename__ = "users"

    id = Column(UUID(as_uuid=False), primary_key=True, comment="用户ID")
    identifier = Column(String(255), unique=True, nullable=False, comment="用户标识符")
    metadata_ = Column(JSON, name="metadata", nullable=False, default={}, comment="用户元数据")
    createdAt = Column(String(50), name="createdAt", comment="创建时间")

    def __repr__(self):
        return f"<User(id='{self.id}', identifier='{self.identifier}', createdAt='{self.createdAt}')>"
