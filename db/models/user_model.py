from datetime import datetime
from typing import Optional

from pydantic import BaseModel, validator
from sqlalchemy import Column, DateTime, Integer, String, Boolean
import hashlib

from db.models.base import BaseModel as BaseBaseModel
from db.base import Base


class UserModel(Base, BaseBaseModel):
    """
    用户模型
    """

    __tablename__ = "user"
    username = Column(String(50), unique=True, index=True, comment="用户名")
    email = Column(String(100), unique=True, index=True, comment="邮箱")
    hashed_password = Column(String(255), comment="哈希密码")
    is_active = Column(Boolean, default=True, comment="是否激活")
    is_superuser = Column(Boolean, default=False, comment="是否为超级用户")

    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}', email='{self.email}', is_active='{self.is_active}')>"


# 创建对应的 Pydantic 模型
class UserSchema(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_superuser: bool
    create_time: Optional[datetime]
    update_time: Optional[datetime]

    class Config:
        from_attributes = True  # 确保可以从 ORM 实例进行验证


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str

    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v

    @validator('password')
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters')
        return v

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v


class UserLogin(BaseModel):
    email: str
    password: str


class UserInDB(UserSchema):
    hashed_password: str