import json
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker, Session
from settings import Settings


SQLALCHEMY_DATABASE_URI = Settings.basic_settings.SQLALCHEMY_DATABASE_URI

# 处理数据库URI，生成异步和同步版本
def get_async_uri(uri: str) -> str:
    """将URI转换为异步版本"""
    if uri.startswith("postgresql://"):
        return uri.replace("postgresql://", "postgresql+asyncpg://")
    elif uri.startswith("postgresql+psycopg2://"):
        return uri.replace("postgresql+psycopg2://", "postgresql+asyncpg://")
    return uri

def get_sync_uri(uri: str) -> str:
    """将URI转换为同步版本"""
    if uri.startswith("postgresql://"):
        return uri.replace("postgresql://", "postgresql+psycopg2://")
    elif uri.startswith("postgresql+asyncpg://"):
        return uri.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    return uri

# 双引擎架构：异步引擎用于DDL操作，同步引擎用于日常ORM操作
# 异步引擎（用于DDL操作如create_all）
async_engine = create_async_engine(
    get_async_uri(SQLALCHEMY_DATABASE_URI),
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False
)

# 同步引擎（用于日常ORM操作）
engine = create_engine(
    get_sync_uri(SQLALCHEMY_DATABASE_URI),
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False
)

# 创建同步会话工厂（用于日常ORM操作）
SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False
)

Base: DeclarativeMeta = declarative_base()

# 在这里导入所有模型，以便 Base.metadata 能够识别它们
def import_models():
    from db.models.user_model import UserModel
    from db.models.thread_model import ThreadModel
    from db.models.step_model import StepModel
    from db.models.element_model import ElementModel
    from db.models.feedback_model import FeedbackModel
    from db.models.message_model import MessageModel
    from db.models.knowledge_base_model import KnowledgeBaseModel
    from db.models.knowledge_file_model import KnowledgeFileModel, FileDocModel
    from db.models.user_memory_model import UserMemoryModel

