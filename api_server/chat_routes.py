from fastapi import APIRouter
from fastapi.params import Depends

from chat.auth_middleware import get_current_user
from chat.kb_chat import kb_chat, chat_process
from utils import build_logger
logger = build_logger()

chat_router = APIRouter(prefix="/api", tags=["ChatChat 对话"])

chat_router.post("/kb_chat", summary="知识库对话", dependencies=[Depends(get_current_user)])(kb_chat)

chat_router.post("/chat", summary="普通对话", dependencies=[Depends(get_current_user)])(chat_process)