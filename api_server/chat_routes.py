from fastapi import APIRouter

from chat.kb_chat import kb_chat
from utils import build_logger
logger = build_logger()

chat_router = APIRouter(prefix="/chat", tags=["ChatChat 对话"])

chat_router.post("/kb_chat", summary="知识库对话")(kb_chat)
