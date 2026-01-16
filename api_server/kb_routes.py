from typing import List
from fastapi import APIRouter, Request
from fastapi.params import Depends

from chat.auth_middleware import get_current_user
from chat.kb_chat import search_docs
from knowledge_base.kb_api import create_kb, list_kbs, delete_kb, list_files
from knowledge_base.kb_doc_api import delete_docs, update_docs, upload_docs, create_memory, get_user_profile
from utils import BaseResponse, ListResponse

kb_router = APIRouter(prefix="/api", tags=["Knowledge Base Management"])

kb_router.get(
    "/list_knowledge_bases", response_model=ListResponse, summary="获取知识库列表", dependencies=[Depends(get_current_user)]
)(list_kbs)

kb_router.post(
    "/create_knowledge_base", response_model=BaseResponse, summary="创建知识库", dependencies=[Depends(get_current_user)]
)(create_kb)

kb_router.post(
    "/delete_knowledge_base", response_model=BaseResponse, summary="删除知识库", dependencies=[Depends(get_current_user)]
)(delete_kb)

kb_router.get(
    "/list_files", response_model=ListResponse, summary="获取知识库内的文件列表", dependencies=[Depends(get_current_user)]
)(list_files)

kb_router.post("/search_docs", response_model=List[dict], summary="搜索知识库", dependencies=[Depends(get_current_user)])(
    search_docs
)

kb_router.post(
    "/upload_docs",
    response_model=BaseResponse,
    summary="上传文件到知识库，并/或进行向量化",
    # dependencies=[Depends(get_current_user)],
)(upload_docs)


kb_router.post(
    "/delete_docs", response_model=BaseResponse, summary="删除知识库内指定文件", dependencies=[Depends(get_current_user)]
)(delete_docs)


kb_router.post(
    "/update_docs", response_model=BaseResponse, summary="更新现有文件到知识库", dependencies=[Depends(get_current_user)]
)(update_docs)


kb_router.post("/create_memory", response_model=BaseResponse, summary="创建记忆")(create_memory)

kb_router.post("/get_user_profile", summary="对回复打分")(get_user_profile)