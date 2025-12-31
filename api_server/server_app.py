from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse

from api_server.chat_routes import chat_router
from api_server.kb_routes import kb_router
from settings import Settings
from starlette.responses import RedirectResponse
from api_server.basic_routes import basicRouter
from chat.auth_middleware import get_current_active_user
from fastapi import Depends, HTTPException, status
from starlette.requests import Request
from starlette.responses import JSONResponse


class AuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        request = Request(scope)
        path = request.url.path

        # 定义不需要认证的路径
        public_paths = [
            "/",
            "/docs",
            "/redoc",
            "/openapi.json",
            # 认证相关路径
            "/api/get_captcha",
            "/api/verify_captcha", 
            "/api/send_email_verification",
            "/api/register",
            "/api/login",
            "/api/refresh_token",
            "/api/config",
            "/api/session",
            "/api/verify",
            "/api/test_chroma",
        ]

        # 检查路径是否需要认证
        requires_auth = not any(path.startswith(p) for p in public_paths)

        if requires_auth:
            # 如果路径需要认证，验证token
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                response = JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"code": 401, "msg": "需要认证", "data": None}
                )
                await response(scope, receive, send)
                return

            token = auth_header[7:]  # 移除 "Bearer " 前缀
            from chat.token_manager import TokenManager
            token_data = TokenManager.verify_token(token)
            
            if token_data is None:
                response = JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"code": 401, "msg": "无效的认证令牌", "data": None}
                )
                await response(scope, receive, send)
                return

        # 继续处理请求
        return await self.app(scope, receive, send)


def create_app():
    app = FastAPI(title="api server")
    
    # 添加认证中间件
    app.add_middleware(AuthMiddleware)
    
    # MakeFastAPIOffline(app)
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if Settings.basic_settings.OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.get("/", summary="swagger 文档", include_in_schema=False)
    async def document():
        return RedirectResponse(url="/docs")

    app.include_router(chat_router)
    app.include_router(kb_router)
    app.include_router(basicRouter)
    return app

def run_api(host, port, **kwargs):
    if kwargs.get("ssl_keyfile") and kwargs.get("ssl_certfile"):
        uvicorn.run(
            app,
            host=host,
            port=port,
            ssl_keyfile=kwargs.get("ssl_keyfile"),
            ssl_certfile=kwargs.get("ssl_certfile"),
        )
    else:
        uvicorn.run(app, host=host, port=port)
app = create_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="langchain-ChatGLM",
        description="About langchain-ChatGLM, local knowledge based ChatGLM with langchain"
        " ｜ 基于本地知识库的 ChatGLM 问答",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--ssl_keyfile", type=str)
    parser.add_argument("--ssl_certfile", type=str)
    # 初始化消息
    args = parser.parse_args()
    args_dict = vars(args)

    run_api(
        host=args.host,
        port=args.port,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
    )