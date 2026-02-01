from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, func
from sqlalchemy.dialects.postgresql import UUID

from db.base import Base


class KnowledgeFileModel(Base):
    """
    知识文件模型
    """

    __tablename__ = "knowledge_file"
    id = Column(UUID(as_uuid=False), primary_key=True, comment="知识文件ID")
    fileName = Column(String(255), name="fileName", comment="文件名")
    fileExt = Column(String(10), name="fileExt", comment="文件扩展名")
    kbName = Column(String(50), name="kbName", comment="所属知识库名称")
    documentLoaderName = Column(String(50), name="documentLoaderName", comment="文档加载器名称")
    textSplitterName = Column(String(50), name="textSplitterName", comment="文本分割器名称")
    fileVersion = Column(Integer, name="fileVersion", default=1, comment="文件版本")
    fileMtime = Column(Float, name="fileMtime", default=0.0, comment="文件修改时间")
    fileSize = Column(Integer, name="fileSize", default=0, comment="文件大小")
    customDocs = Column(Boolean, name="customDocs", default=False, comment="是否自定义docs")
    docsCount = Column(Integer, name="docsCount", default=0, comment="切分文档数量")
    createdAt = Column(String(50), name="createdAt", default=func.now(), comment="创建时间")

    def __repr__(self):
        return f"<KnowledgeFile(id='{self.id}', fileName='{self.fileName}', fileExt='{self.fileExt}', kbName='{self.kbName}', documentLoaderName='{self.documentLoaderName}', textSplitterName='{self.textSplitterName}', fileVersion='{self.fileVersion}', createdAt='{self.createdAt}')>"


class FileDocModel(Base):
    """
    文件-向量库文档模型
    """

    __tablename__ = "file_doc"
    id = Column(UUID(as_uuid=False), primary_key=True, comment="ID")
    kbName = Column(String(50), name="kbName", comment="知识库名称")
    fileName = Column(String(255), name="fileName", comment="文件名称")
    docId = Column(String(50), name="docId", comment="向量库文档ID")
    metadata_ = Column(JSON, name="metadata", default={})

    def __repr__(self):
        return f"<FileDoc(id='{self.id}', kbName='{self.kbName}', fileName='{self.fileName}', docId='{self.docId}', metadata='{self.metadata_}')>"
