import json
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker
from settings import Settings


SQLALCHEMY_DATABASE_URI = Settings.basic_settings.SQLALCHEMY_DATABASE_URI

# 确保 URI 使用 asyncpg 协议
if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgresql://"):
    SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgresql://", "postgresql+asyncpg://")

# 创建异步引擎
engine = create_async_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
    # 针对 PostgreSQL 的连接池配置
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=False  # 设置为 True 可以看到 SQL 日志
)

# 创建异步会话工厂
SessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
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

