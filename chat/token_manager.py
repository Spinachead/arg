import jwt
import datetime
import os
from typing import Optional
from fastapi import HTTPException, status
from pydantic import BaseModel
from db.models.user_model import UserSchema
from dotenv import load_dotenv
load_dotenv()

class TokenData(BaseModel):
    user_id: Optional[int] = None
    email: Optional[str] = None


class TokenManager:
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this-in-production")  # 从环境变量获取密钥
    ALGORITHM = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    @classmethod
    def create_access_token(cls, data: dict, expires_delta: Optional[datetime.timedelta] = None):
        """
        创建访问令牌
        :param data: 要编码到token中的数据
        :param expires_delta: 过期时间
        :return: JWT token字符串
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.datetime.now(datetime.timezone.utc) + expires_delta
        else:
            expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
                minutes=cls.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, cls.SECRET_KEY, algorithm=cls.ALGORITHM)
        return encoded_jwt

    @classmethod
    def create_refresh_token(cls, data: dict):
        """
        创建刷新令牌
        :param data: 要编码到token中的数据
        :return: JWT token字符串
        """
        to_encode = data.copy()
        expire = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(
            days=cls.REFRESH_TOKEN_EXPIRE_DAYS
        )
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, cls.SECRET_KEY, algorithm=cls.ALGORITHM)
        return encoded_jwt

    @classmethod
    def verify_token(cls, token: str) -> Optional[TokenData]:
        """
        验证token并返回用户信息
        :param token: JWT token字符串
        :return: TokenData对象或None
        """
        try:
            payload = jwt.decode(token, cls.SECRET_KEY, algorithms=[cls.ALGORITHM])
            user_id: int = payload.get("user_id")
            email: str = payload.get("email")
            token_type: str = payload.get("type")
            
            if user_id is None or email is None or token_type != "access":
                return None
            
            token_data = TokenData(user_id=user_id, email=email)
            return token_data
        except jwt.PyJWTError:
            return None

    @classmethod
    def verify_refresh_token(cls, token: str) -> Optional[TokenData]:
        """
        验证刷新token并返回用户信息
        :param token: JWT refresh token字符串
        :return: TokenData对象或None
        """
        try:
            payload = jwt.decode(token, cls.SECRET_KEY, algorithms=[cls.ALGORITHM])
            user_id: int = payload.get("user_id")
            email: str = payload.get("email")
            token_type: str = payload.get("type")
            
            if user_id is None or email is None or token_type != "refresh":
                return None
            
            token_data = TokenData(user_id=user_id, email=email)
            return token_data
        except jwt.PyJWTError:
            return None