from fastapi import APIRouter

from chat.kb_chat import kb_chat, chat_process
from utils import build_logger
from chat.auth_middleware import get_current_active_user
from fastapi import Depends

logger = build_logger()

chat_router = APIRouter(prefix="/api", tags=["ChatChat 对话"])

# 为聊天接口添加认证依赖
chat_router.post("/kb_chat", summary="知识库对话", dependencies=[Depends(get_current_active_user)])(kb_chat)

chat_router.post("/chat", summary="普通对话", dependencies=[Depends(get_current_active_user)])(chat_process)