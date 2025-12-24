import os
from knowledge_base.utils import get_file_path, KnowledgeFile, files2docs_in_thread, get_kb_path

os.environ["OTEL_SDK_DISABLED"] = "true"
import uuid
from datetime import datetime

from fastapi import FastAPI, Request, Body, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse
from typing import Optional, Dict, Any, AsyncIterable, Literal, List, TypedDict, Annotated
import asyncio
import json

from knowledge_base.kb_service.base import KBServiceFactory, get_kb_file_details
from rag_chain import create_rag_graph
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from utils import format_reference, wrap_done, get_prompt_template, History, run_in_thread_pool, \
    get_default_embedding, BaseResponse, ListResponse
# 在导入语句之后，FastAPI应用创建之前添加
from db.base import Base, engine
from utils import build_logger
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = build_logger()
# 确保在所有模型导入之后调用下面的方法
Base.metadata.create_all(bind=engine)

load_dotenv()

# 导入 is_not_empty_string 函数

# 确保 Ollama 正在运行！
app = FastAPI(
    title="RAG API Service",
    version="1.0",
    description="基于 Qwen + 本地文档的问答 API"
)


if __name__ == "__main__":
    import uvicorn
    # 绑定到 localhost 只允许本地访问
    get_kb_path("samples")
    uvicorn.run(app, host="localhost", port=8000)
