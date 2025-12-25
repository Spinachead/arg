from fastapi import APIRouter

from utils import build_logger
from chat.basic import config, session, verify, test_chroma

logger = build_logger()

basicRouter = APIRouter(prefix="/api", tags=["basic路由"])

basicRouter.post("/config", summary="获取配置信息")(config)

basicRouter.post("/session", summary="获取session")(session)

basicRouter.post("/verify", summary="验证")(verify)

basicRouter.get("/test_chroma", summary="测试chroma")(test_chroma)