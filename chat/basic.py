import os

from fastapi import FastAPI, Request, Body, UploadFile, File, Form, Query

from knowledge_base.kb_service.base import KBServiceFactory
from utils import BaseResponse


async def config()->BaseResponse:
    return BaseResponse(code=200, msg="成功", data= {'apiModel': 'Mock-API', 'socksProxy': False, 'httpsProxy': False})

async def session(req: Request)->BaseResponse:
    try:
        auth_secret_key = os.getenv("AUTH_SECRET_KEY")
        has_auth = auth_secret_key and len(auth_secret_key) > 0
        return BaseResponse(code=200, msg="成功", data={'auth': has_auth, 'model': 'default'})

    except Exception as e:
        return BaseResponse(code=200, msg="失败", data=None)


async def verify(req: Request):
    try:
        body = await req.json()
        token = body.get("token")

        if not token:
            raise ValueError('Secret key is empty')

        auth_secret_key = os.getenv("AUTH_SECRET_KEY")
        if auth_secret_key != token:
            raise ValueError('密钥无效 | Secret key is invalid')

        return BaseResponse(code=200, msg="Verify successfully", data=None)
    except Exception as e:
        return BaseResponse(code=200, msg="fail", data=None)


def test_chroma(
        knowledge_base_name: str = Query(
            ..., description="知识库名称", examples=["samples"]
        ),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        query: str = Query(..., description="查询内容"),
        score_threshold: float = Query(
            None, description="相似度阈值", examples=[0.5]
        ),

)-> BaseResponse:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    # data = kb.list_docs(file_name="体检报告.pdf")
    data = kb.search_docs(query, top_k=3, score_threshold=score_threshold)
    return BaseResponse(code=200, msg="成功", data=data)