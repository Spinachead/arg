import os
import re
from fastapi import FastAPI, Request, Body, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from db.session import SessionLocal
from db.models.user_model import UserCreate, UserLogin
from db.repository.user_repository import get_user_by_email, create_user, verify_password, hash_password
from utils import generate_verification_code, send_verification_email, generate_captcha, verification_code_cache, captcha_cache
from utils import BaseResponse


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_captcha():
    """获取图形验证码"""
    text, image_base64 = generate_captcha()
    # 生成一个唯一的key用于存储验证码
    import uuid
    captcha_key = f"captcha_{uuid.uuid4().hex}"
    captcha_cache.set(captcha_key, text, expire_in_seconds=300)  # 5分钟过期
    
    return {
        "status": "Success",
        "message": "验证码获取成功",
        "data": {
            "captcha_key": captcha_key,
            "captcha_image": f"data:image/png;base64,{image_base64}"
        }
    }


async def verify_captcha(captcha_key: str = Body(...), captcha_code: str = Body(...)):
    """验证图形验证码"""
    stored_code = captcha_cache.get(captcha_key)
    
    if not stored_code:
        return {"status": "Fail", "message": "验证码已过期", "data": None}
    
    if stored_code.upper() != captcha_code.upper():
        return {"status": "Fail", "message": "验证码错误", "data": None}
    
    # 验证成功后删除该验证码
    captcha_cache.delete(captcha_key)
    
    return {"status": "Success", "message": "验证码验证成功", "data": None}


async def send_email_verification(email: str = Body(...)):
    """发送邮箱验证码"""
    # 验证邮箱格式
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return {"status": "Fail", "message": "邮箱格式不正确", "data": None}
    
    # 生成验证码
    verification_code = generate_verification_code()
    
    # 存储验证码到缓存
    verification_code_cache.set(email, verification_code, expire_in_seconds=300)  # 5分钟过期
    
    # 发送验证码邮件
    success = send_verification_email(email, verification_code)
    
    if not success:
        return {"status": "Fail", "message": "验证码发送失败", "data": None}
    
    return {"status": "Success", "message": "验证码已发送至您的邮箱", "data": None}


async def register(
    email: str = Body(...),
    password: str = Body(...),
    confirm_password: str = Body(...),
    email_verification_code: str = Body(...)
):
    """用户注册"""
    # 验证邮箱格式
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(pattern, email):
        return {"status": "Fail", "message": "邮箱格式不正确", "data": None}
    
    # 验证密码长度
    if len(password) < 6:
        return {"status": "Fail", "message": "密码长度不能少于6位", "data": None}
    
    # 验证密码和确认密码是否一致
    if password != confirm_password:
        return {"status": "Fail", "message": "密码和确认密码不一致", "data": None}
    
    # 验证邮箱验证码
    stored_code = verification_code_cache.get(email)
    if not stored_code:
        return {"status": "Fail", "message": "邮箱验证码已过期", "data": None}
    
    if stored_code != email_verification_code:
        return {"status": "Fail", "message": "邮箱验证码错误", "data": None}
    
    # 检查邮箱是否已被注册
    db = next(get_db())
    try:
        existing_user = get_user_by_email(db, email)
        if existing_user:
            return {"status": "Fail", "message": "该邮箱已被注册", "data": None}
        
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
        
        return {
            "status": "Success", 
            "message": "注册成功", 
            "data": {
                "user_id": user.id,
                "email": user.email
            }
        }
    finally:
        db.close()


async def login(req: Request):
    """用户登录"""
    try:
        body = await req.json()
        email = body.get("email")
        password = body.get("password")
        
        if not email or not password:
            return {"status": "Fail", "message": "邮箱和密码不能为空", "data": None}
        
        # 验证邮箱格式
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            return {"status": "Fail", "message": "邮箱格式不正确", "data": None}
        
        db = next(get_db())
        try:
            user = get_user_by_email(db, email)
            if not user:
                return {"status": "Fail", "message": "用户不存在", "data": None}
            
            if not user.is_active:
                return {"status": "Fail", "message": "账户已被禁用", "data": None}
            
            if not verify_password(password, user.hashed_password):
                return {"status": "Fail", "message": "密码错误", "data": None}
            
            # 登录成功，可以在这里生成token（这里简化处理）
            return {
                "status": "Success",
                "message": "登录成功",
                "data": {
                    "user_id": user.id,
                    "email": user.email,
                    "username": user.username
                }
            }
        finally:
            db.close()
    except Exception as e:
        return {"status": "Fail", "message": str(e), "data": None}