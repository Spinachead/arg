from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, func

from db.base import Base


class KnowledgeBaseModel(Base):
    """
    知识库模型
    """

    __tablename__ = "knowledge_base"
    id = Column(String(36), primary_key=True, comment="知识库ID")
    kbName = Column(String(50), name="kbName", comment="知识库名称")
    kbInfo = Column(String(200), name="kbInfo", comment="知识库简介(用于Agent)")
    vsType = Column(String(50), name="vsType", comment="向量库类型")
    embedModel = Column(String(50), name="embedModel", comment="嵌入模型名称")
    fileCount = Column(Integer, name="fileCount", default=0, comment="文件数量")
    createdAt = Column(String(50), name="createdAt", default=func.now(), comment="创建时间")

    def __repr__(self):
        return f"<KnowledgeBase(id='{self.id}', kbName='{self.kbName}', kbInfo='{self.kbInfo}', vsType='{self.vsType}', embedModel='{self.embedModel}', fileCount='{self.fileCount}', createdAt='{self.createdAt}')>"


# 创建一个对应的 Pydantic 模型
class KnowledgeBaseSchema(BaseModel):
    id: str
    kbName: str
    kbInfo: Optional[str]
    vsType: Optional[str]
    embedModel: Optional[str]
    fileCount: Optional[int]
    createdAt: Optional[str]

    class Config:
        from_attributes = True  # 确保可以从 ORM 实例进行验证
