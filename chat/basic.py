import os

from fastapi import FastAPI, Request, Body, UploadFile, File, Form, Query

from knowledge_base.kb_service.base import KBServiceFactory
from utils import BaseResponse


async def config():
    return {"status": 'Success', "data": {'apiModel': 'Mock-API', 'socksProxy': 'false', 'httpsProxy': 'false'},
            "message": "Success"}


async def session(req: Request):
    try:
        auth_secret_key = os.getenv("AUTH_SECRET_KEY")
        has_auth = auth_secret_key and len(auth_secret_key) > 0
        # 注意：currentModel 函数在原 JS 代码中未定义，这里保持简单实现
        return {
            "status": 'Success',
            "message": "",
            "data": {
                "auth": has_auth,
                "model": "default"
            }
        }
    except Exception as e:
        return {"status": 'Fail', "message": str(e), "data": None}


async def verify(req: Request):
    try:
        body = await req.json()
        token = body.get("token")

        if not token:
            raise ValueError('Secret key is empty')

        auth_secret_key = os.getenv("AUTH_SECRET_KEY")
        if auth_secret_key != token:
            raise ValueError('密钥无效 | Secret key is invalid')

        return {"status": 'Success', "message": "Verify successfully", "data": None}
    except Exception as e:
        return {"status": 'Fail', "message": str(e), "data": None}


def test_chroma(
        knowledge_base_name: str = Query(
            ..., description="知识库名称", examples=["samples"]
        ),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        query: str = Query(..., description="查询内容"),
)-> BaseResponse:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    # data = kb.list_docs(file_name="体检报告.pdf")
    data = kb.search_docs(query)
    return BaseResponse(code=200, msg="成功", data=data)