from fastapi import APIRouter

from utils import build_logger
from chat.basic import config, session, verify, test_chroma
from chat.auth import get_captcha, verify_captcha, send_email_verification, register, login, refresh_token, \
    get_user_info

logger = build_logger()

basicRouter = APIRouter(prefix="/api", tags=["basic路由"])

basicRouter.post("/config", summary="获取配置信息")(config)

basicRouter.post("/session", summary="获取session")(session)

basicRouter.post("/verify", summary="验证")(verify)

basicRouter.get("/test_chroma", summary="测试chroma")(test_chroma)

# 认证相关路由
basicRouter.get("/get_captcha", summary="获取图形验证码")(get_captcha)
basicRouter.post("/verify_captcha", summary="验证图形验证码")(verify_captcha)
basicRouter.post("/send_email_verification", summary="发送邮箱验证码")(send_email_verification)
basicRouter.post("/register", summary="用户注册")(register)
basicRouter.post("/login", summary="用户登录")(login)
basicRouter.post("/refresh_token", summary="刷新访问令牌")(refresh_token)
basicRouter.get("/user_info", summary="获取用户信息")(get_user_info)