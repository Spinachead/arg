import json
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker
from settings import Settings

# 使用同步引擎用于基础表创建和传统 ORM 操作
# 注意：PostgreSQL 推荐使用 psycopg3 或 psycopg2
# 这里从配置中获取连接字符串
SQLALCHEMY_DATABASE_URI = Settings.basic_settings.SQLALCHEMY_DATABASE_URI

# 适配处理：如果 URI 包含 asyncpg 前缀，则将其转换为同步协议，以便 create_engine (sync) 使用
if SQLALCHEMY_DATABASE_URI and SQLALCHEMY_DATABASE_URI.startswith("postgresql+asyncpg://"):
    SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgresql+asyncpg://", "postgresql://")

engine = create_engine(
    SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
    # 针对 PostgreSQL 的连接池配置
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

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

