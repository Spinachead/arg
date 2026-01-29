import json

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker

from settings import Settings
from pathlib import Path

# 确保数据库目录存在
db_path = Path(Settings.basic_settings.DB_ROOT_PATH)
db_path.parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    Settings.basic_settings.SQLALCHEMY_DATABASE_URI,
    json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base: DeclarativeMeta = declarative_base()
