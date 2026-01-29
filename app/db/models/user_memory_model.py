# db/models/user_memory_model.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, func, ForeignKey
from db.base import Base

class UserMemoryModel(Base):
    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("user.id"), index=True, comment="用户ID")
    memory_text = Column(String(1024), comment="记忆内容摘要")
    importance = Column(Integer, default=1, comment="重要程度 1-5")
    last_used_time = Column(DateTime, default=func.now(), comment="最近使用时间")
    create_time = Column(DateTime, default=func.now(), comment="创建时间")
    meta_data = Column(JSON, default={})