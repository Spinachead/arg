from sqlalchemy import Column, String, JSON, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from db.base import Base

class ElementModel(Base):
    """
    元素模型 (Chainlit Elements)
    """
    __tablename__ = "elements"

    id = Column(UUID(as_uuid=False), primary_key=True, comment="元素ID")
    threadId = Column(UUID(as_uuid=False), ForeignKey("threads.id", ondelete="CASCADE"), name="threadId", comment="线程ID")
    type = Column(String(50), comment="元素类型")
    url = Column(String(1024), comment="URL")
    chainlitKey = Column(String(255), name="chainlitKey", comment="Chainlit Key")
    name = Column(String(255), nullable=False, comment="元素名称")
    display = Column(String(50), comment="显示方式")
    objectKey = Column(String(255), name="objectKey", comment="对象Key")
    size = Column(String(50), comment="大小")
    page = Column(Integer, comment="页码")
    language = Column(String(50), comment="语言")
    forId = Column(UUID(as_uuid=False), name="forId", comment="关联ID")
    mime = Column(String(100), comment="MIME类型")
    props = Column(JSON, comment="属性")

    def __repr__(self):
        return f"<Element(id='{self.id}', name='{self.name}', type='{self.type}', threadId='{self.threadId}')>"
