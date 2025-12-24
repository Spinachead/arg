from typing import List
from fastapi import APIRouter, Request
from knowledge_base.kb_api import create_kb, list_kbs, delete_kb, list_files
from knowledge_base.kb_doc_api import delete_docs, update_docs
from server import upload_docs, search_docs
from utils import BaseResponse, ListResponse

kb_router = APIRouter(prefix="/api", tags=["Knowledge Base Management"])

kb_router.get(
    "/list_knowledge_bases", response_model=ListResponse, summary="获取知识库列表"
)(list_kbs)

kb_router.post(
    "/create_knowledge_base", response_model=BaseResponse, summary="创建知识库"
)(create_kb)

kb_router.post(
    "/delete_knowledge_base", response_model=BaseResponse, summary="删除知识库"
)(delete_kb)

kb_router.get(
    "/list_files", response_model=ListResponse, summary="获取知识库内的文件列表"
)(list_files)

kb_router.post("/search_docs", response_model=List[dict], summary="搜索知识库")(
    search_docs
)

kb_router.post(
    "/upload_docs",
    response_model=BaseResponse,
    summary="上传文件到知识库，并/或进行向量化",
)(upload_docs)


kb_router.post(
    "/delete_docs", response_model=BaseResponse, summary="删除知识库内指定文件"
)(delete_docs)


kb_router.post(
    "/update_docs", response_model=BaseResponse, summary="更新现有文件到知识库"
)(update_docs)