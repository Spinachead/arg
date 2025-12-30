import os
import re
from typing import Any, Coroutine

from fastapi import FastAPI, Request, Body, Query, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models.user_model import UserCreate, UserLogin
from db.repository.user_repository import get_user_by_email, create_user, verify_password, hash_password
from captcha import generate_verification_code, send_verification_email, generate_captcha, verification_code_cache, captcha_cache
from utils import BaseResponse
from .token_manager import TokenManager, TokenData
import datetime
from .auth_middleware import get_current_active_user


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_captcha()->BaseResponse:
    """获取图形验证码"""
    text, image_base64 = generate_captcha()
    # 生成一个唯一的key用于存储验证码
    import uuid
    captcha_key = f"captcha_{uuid.uuid4().hex}"
    captcha_cache.set(captcha_key, text, expire_in_seconds=300)  # 5分钟过期
    return BaseResponse(code=200, msg="成功", data = {"captcha_key": captcha_key, "captcha_image": f"data:image/png;base64,{image_base64}"})


async def verify_captcha(captcha_key: str = Body(...), captcha_code: str = Body(...))->BaseResponse:
    """验证图形验证码"""
    stored_code = captcha_cache.get(captcha_key)
    
    if not stored_code:
        return BaseResponse(code=400, msg="验证码已过期", data=None)

    if stored_code.upper() != captcha_code.upper():
        return BaseResponse(code=400, msg="验证码错误", data=None)

    # 验证成功后删除该验证码
    captcha_cache.delete(captcha_key)
    return  BaseResponse(code=200, msg="验证码验证成功", data=None)


async def send_email_verification(email: str = Body(..., embed=True))->BaseResponse:
    """发送邮箱验证码"""
    # 验证邮箱格式
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return BaseResponse(code=400, msg="邮箱格式不正确", data=None)

    # 生成验证码
    verification_code = generate_verification_code()
    
    # 存储验证码到缓存
    verification_code_cache.set(email, verification_code, expire_in_seconds=300)  # 5分钟过期
    
    # 发送验证码邮件
    success = send_verification_email(email, verification_code)
    
    if not success:
        return BaseResponse(code=200, msg="验证码发送失败", data=None)
    
    return BaseResponse(code=200, msg="验证码已发送至您的邮箱", data=None)


async def register(
    email: str = Body(...),
    password: str = Body(...),
    confirm_password: str = Body(...),
    email_verification_code: str = Body(...)
)-> BaseResponse:
    """用户注册"""
    # 验证邮箱格式
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return BaseResponse(code=400, msg="邮箱格式不正确", data=None)
    
    # 验证密码长度
    if len(password) < 6:
        return BaseResponse(code=400, msg="密码长度不能少于6位", data=None)
    
    # 验证密码和确认密码是否一致
    if password != confirm_password:
        return BaseResponse(code=400, msg="密码和确认密码不一致", data=None)
    
    # 验证邮箱验证码
    stored_code = verification_code_cache.get(email)
    if not stored_code:
        return BaseResponse(code=400, msg="邮箱验证码已过期", data=None)
    
    if stored_code != email_verification_code:
        return BaseResponse(code=400, msg="邮箱验证码错误", data=None)
    
    # 检查邮箱是否已被注册
    db = next(get_db())
    try:
        existing_user = get_user_by_email(db, email)
        if existing_user:
            return BaseResponse(code=200, msg="该邮箱已被注册", data=None)
        
        # 创建新用户
        user_create = UserCreate(
            username=email.split('@')[0],  # 使用邮箱前缀作为用户名
            email=email,
            password=password,
            confirm_password=confirm_password
        )
        
        hashed_password = hash_password(password)
        user = create_user(db, user_create, hashed_password)
        
        # 注册成功后删除验证码
        verification_code_cache.delete(email)
        
        # 生成访问令牌
        access_token_data = {
            "user_id": user.id,
            "email": user.email
        }
        access_token = TokenManager.create_access_token(data=access_token_data)
        refresh_token = TokenManager.create_refresh_token(data=access_token_data)
        
        return BaseResponse(code=200, msg="注册成功", 
                            data={
                                'user_id': user.id, 
                                'email': user.email,
                                'username': user.username,
                                'access_token': access_token,
                                'refresh_token': refresh_token,
                                'token_type': 'bearer'
                            })
    finally:
        db.close()


async def login(
        email: str = Body(...),
        password: str = Body(...)
)-> BaseResponse:
    """用户登录"""
    if not email or not password:
        return BaseResponse(code=400, msg="邮箱和密码不能为空", data=None)

    # 验证邮箱格式
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return BaseResponse(code=400, msg="邮箱格式不正确", data=None)

    db = next(get_db())
    try:
        user = get_user_by_email(db, email)
        if not user:
            return BaseResponse(code=400, msg="用户不存在", data=None)

        if not user.is_active:
            return BaseResponse(code=400, msg="账户已被禁用", data=None)

        if not verify_password(password, user.hashed_password):
            return BaseResponse(code=400, msg="密码错误", data=None)

        # 登录成功，生成token
        access_token_data = {
            "user_id": user.id,
            "email": user.email
        }
        access_token = TokenManager.create_access_token(data=access_token_data)
        refresh_token = TokenManager.create_refresh_token(data=access_token_data)
        
        return BaseResponse(code=200, msg="登录成功",
                            data={
                                'user_id': user.id, 
                                'email': user.email, 
                                'username': user.username,
                                'access_token': access_token,
                                'refresh_token': refresh_token,
                                'token_type': 'bearer'
                            })
    finally:
        db.close()


async def refresh_token(
        refresh_token: str = Body(..., embed=True)
)-> BaseResponse:
    """使用刷新令牌获取新的访问令牌"""
    token_data = TokenManager.verify_refresh_token(refresh_token)
    
    if token_data is None:
        return BaseResponse(code=400, msg="无效的刷新令牌", data=None)
    
    # 生成新的访问令牌
    new_access_token_data = {
        "user_id": token_data.user_id,
        "email": token_data.email
    }
    new_access_token = TokenManager.create_access_token(data=new_access_token_data)
    
    return BaseResponse(code=200, msg="令牌刷新成功",
                        data={
                            'access_token': new_access_token,
                            'token_type': 'bearer'
                        })


async def get_user_info(current_user: TokenData = Depends(get_current_active_user)):
    """获取当前用户信息 - 需要认证的端点示例"""
    return BaseResponse(code=200, msg="获取用户信息成功", 
                        data={
                            'user_id': current_user.user_id,
                            'email': current_user.email
                        })