from sqlalchemy import Column, String, JSON, Boolean, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from db.base import Base

class StepModel(Base):
    """
    步骤模型 (Chainlit Steps)
    """
    __tablename__ = "steps"

    id = Column(UUID(as_uuid=False), primary_key=True, comment="步骤ID")
    name = Column(String(255), nullable=False, comment="步骤名称")
    type = Column(String(50), nullable=False, comment="步骤类型")
    threadId = Column(UUID(as_uuid=False), ForeignKey("threads.id", ondelete="CASCADE"), name="threadId", nullable=False, comment="线程ID")
    parentId = Column(UUID(as_uuid=False), name="parentId", comment="父步骤ID")
    streaming = Column(Boolean, nullable=False, default=False, comment="是否流式输出")
    waitForAnswer = Column(Boolean, name="waitForAnswer", comment="是否等待回答")
    isError = Column(Boolean, name="isError", comment="是否出错")
    metadata_ = Column(JSON, name="metadata", comment="元数据")
    tags = Column(JSON, comment="标签")
    input = Column(String, comment="输入内容")
    output = Column(String, comment="输出内容")
    createdAt = Column(String(50), name="createdAt", comment="创建时间")
    command = Column(String(255), comment="命令")
    start = Column(String(50), comment="开始时间")
    end = Column(String(50), comment="结束时间")
    generation = Column(JSON, comment="生成信息")
    showInput = Column(String(255), name="showInput", comment="是否显示输入")
    language = Column(String(50), comment="语言")
    indent = Column(Integer, comment="缩进")
    defaultOpen = Column(Boolean, name="defaultOpen", comment="默认展开")

    def __repr__(self):
        return f"<Step(id='{self.id}', name='{self.name}', type='{self.type}', threadId='{self.threadId}')>"
