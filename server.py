import os

os.environ["OTEL_SDK_DISABLED"] = "true"
# server.py
import pprint
import uuid
from datetime import datetime

from fastapi import FastAPI, Request, Body
from fastapi.responses import StreamingResponse
from langchain_classic.callbacks import AsyncIteratorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncIterable, Literal, List
import asyncio
import json

from api_schemas import OpenAIChatOutput
from knowledge_base.kb_service.base import KBServiceFactory
from knowledge_base.model.kb_document_model import DocumentWithVSId
from rag_chain import create_rag_graph
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
from sse_starlette.sse import EventSourceResponse

from utils import format_reference, get_ChatOpenAI, wrap_done, get_prompt_template, History
# 在导入语句之后，FastAPI应用创建之前添加
from db.base import Base, engine
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

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatProcessRequest(BaseModel):
    prompt: str
    options: Optional[Dict[str, Any]] = {}
    systemMessage: Optional[str] = ""
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


# 创建 RAG Chain（首次会构建向量库）
rag_chain = create_rag_graph()

# 挂载到 /rag 路径
add_routes(
    app,
    rag_chain,
    path="/api"
)


@app.get("/")
async def root():
    return {"message": "RAG Server is running. Visit /docs for API documentation or /rag/playground for playground."}


@app.post("/api/chat")
async def chat_process(request: ChatProcessRequest):
    if not request.prompt:
        return {"status": "Error", "data": None, "message": "Question input is required"}

    try:
        # 获取会话ID，用于记忆功能
        session_id = request.options.get("sessionId", "default_session") if request.options else "default_session"
        config = {"configurable": {"thread_id": session_id}}

        # 使用带记忆功能的 RAG 链处理请求
        input_data = {
            "messages": [HumanMessage(content=request.prompt)]
        }
        response = rag_chain.invoke(input_data, config=config)

        return {
            "status": "Success",
            "data": {
                "id": "chat-1",
                "role": "assistant",
                "text": response["messages"][-1].content,
                "dateTime": "1111111"
            },
            "message": "Success"
        }
    except Exception as e:
        return {"status": "Error", "data": None, "message": str(e)}


@app.post("/api/chat-process")
async def chat_process_stream(request: ChatProcessRequest):
    async def generate_response() -> AsyncIterable[str]:
        try:
            if not request.prompt:
                error_data = {
                    'status': 'Error',
                    'data': None,
                    'message': 'Question input is required'
                }
                yield f"{json.dumps(error_data)}\n"
                return

            # 获取会话ID，用于记忆功能
            session_id = request.options.get("sessionId", "default_session") if request.options else "default_session"
            config = {"configurable": {"thread_id": session_id}}

            # 使用带记忆功能的 RAG 链处理请求
            input_data = {
                "messages": [HumanMessage(content=request.prompt)]
            }
            response = rag_chain.invoke(input_data, config=config)
            result = response["messages"][-1].content

            # 流式输出，按字符逐个输出
            for i, char in enumerate(result):
                chat_data = {
                    "id": "chat-1",
                    "role": "assistant",
                    "text": result[:i + 1],
                    "dateTime": "1111111"
                }
                yield f"{json.dumps(chat_data)}\n"
                await asyncio.sleep(0.01)  # 控制输出速度

        except Exception as e:
            error_data = {
                'status': 'Error',
                'data': None,
                'message': str(e)
            }
            yield f"{json.dumps(error_data)}\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.post("/api/chat-process2")
async def chat_process_stream(request: ChatProcessRequest):
    async def generate_response() -> AsyncIterable[str]:
        try:
            if not request.prompt:
                error_data = {
                    'status': 'Error',
                    'data': None,
                    'message': 'Question input is required'
                }
                yield f"{json.dumps(error_data)}\n"
                return

            # 获取会话ID，用于记忆功能
            session_id = request.options.get("sessionId", "default_session") if request.options else "default_session"
            config = {"configurable": {"thread_id": session_id}}

            # 使用带记忆功能的 RAG 链处理请求
            input_data = {
                "messages": [HumanMessage(content=request.prompt)]
            }

            # 使用真正的流式方法
            full_content = ""
            for chunk in rag_chain.stream(input_data, config=config, stream_mode="messages"):
                # chunk 是一个元组 (step, data)
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    step, data = chunk
                else:
                    # 兼容不同的返回格式
                    data = chunk

                # 根据实际返回的数据结构调整访问方式
                content = ""
                if isinstance(data, dict):
                    if "messages" in data and len(data["messages"]) > 0:
                        # 如果是消息列表，获取最后一个消息的内容
                        message = data["messages"][-1]
                        content = getattr(message, 'content', str(message))
                    elif "content" in data:
                        # 如果直接包含content字段
                        content = data["content"]

                # 如果有新内容，则处理并发送
                if content and content != full_content:
                    # 获取新增的内容
                    new_content = content[len(full_content):]
                    full_content = content

                    # 逐字符发送新增内容，实现打字机效果
                    for char in new_content:
                        chat_data = {
                            "id": "chat-1",
                            "conversationId": session_id,
                            "text": full_content[:len(full_content) - len(new_content) + 1],
                            "dateTime": datetime.now().isoformat(),
                            "detail": {
                                "choices": [{
                                    "finish_reason": None
                                }]
                            }
                        }
                        yield f"{json.dumps(chat_data)}\n"
                        await asyncio.sleep(0.01)  # 控制打字机效果的速度

            # 发送最终完成的消息
            if full_content:
                chat_data = {
                    "id": "chat-1",
                    "conversationId": session_id,
                    "text": full_content,
                    "dateTime": datetime.now().isoformat(),
                    "detail": {
                        "choices": [{
                            "finish_reason": "stop"
                        }]
                    }
                }
                yield f"{json.dumps(chat_data)}\n"

        except Exception as e:
            error_data = {
                'status': 'Error',
                'data': None,
                'message': str(e)
            }
            print(error_data)
            yield f"{json.dumps(error_data)}\n"

    return StreamingResponse(generate_response(), media_type="text/event-stream")


@app.post("/api/config")
async def config():
    return {"status": 'Success', "data": {'apiModel': 'Mock-API', 'socksProxy': 'false', 'httpsProxy': 'false'},
            "message": "Success"}


@app.post("/api/session")
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


@app.post("/api/verify")
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


def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(
            ..., description="知识库名称", examples=["samples"]
        ),
        top_k: int = Body(3, description="匹配向量数"),
        score_threshold: float = Body(
            2.0,
            description="知识库匹配相关度阈值，取值范围在0-1之间，"
                        "SCORE越小，相关度越高，"
                        "取到2相当于不筛选，建议设置在0.5左右",
            ge=0.0,
            le=2.0,
        ),
        file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[Dict]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k, score_threshold)
            # data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
            data = [DocumentWithVSId(**{"id": x.metadata.get("id"), **x.dict()}) for x in docs]
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
    return [x.dict() for x in data]


async def kb_chat(query: str = Body(..., description="用户输入", example=["你好"]),
                  mode: Literal["local_kb"] = Body("local_kb", description="知识来源"),
                  top_k: int = Body(3, description="匹配向量数字"),
                  score_threshold: float = Body(
                      2.0,
                      description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                      ge=0,
                      le=2,
                  ),
                  history: List[History] = Body(
                      [],
                      description="历史对话",
                      examples=[[
                          {"role": "user",
                           "content": "我们来玩成语接龙，我先来，生龙活虎"},
                          {"role": "assistant",
                           "content": "虎头虎脑"}]]
                  ),
                  kb_name: str = Body("",
                                      description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称",
                                      examples=["samples"]),

                  stream: bool = Body(True, description="流式输出"),
                  model: str = Body("qwen:1.8b", description="LLM 模型名称。"),
                  temperature: float = Body(0.7, description="LLM 采样温度", ge=0.0,
                                            le=2.0),
                  max_tokens: Optional[int] = Body(
                      None,
                      description="限制LLM生成Token数量，默认None代表模型最大值"
                  ),
                  prompt_name: str = Body(
                      "default",
                      description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                  ),
                  return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
                  request: Request = None,
                  ):
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            nonlocal prompt_name, max_tokens
            docs = search_docs(query=query,
                               knowledge_base_name=kb_name,
                               top_k=top_k,
                               score_threshold=score_threshold,
                               file_name="",
                               metadata={})

            source_documents = format_reference(kb_name, docs, "")
            if return_direct:
                yield OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    model=None,
                    object="chat.completion",
                    content="",
                    role="assistant",
                    finish_reason="stop",
                    docs=source_documents,
                ).model_dump_json()
                return

            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]

            llm = get_ChatOpenAI(
                model_name=model,
                temperature=temperature,
                max_tokens=None,
                callbacks=callbacks,
            )

            context = "\n\n".join([doc["page_content"] for doc in docs])

            if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
                prompt_name = "empty"
            prompt_template = get_prompt_template("rag", prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])

            chain = chat_prompt | llm

            # Begin a task that runs in the background.
            task = asyncio.create_task(wrap_done(
                chain.ainvoke({"context": context, "question": query}),
                callback.done),
            )

            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

            if stream:
                # yield documents first
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion.chunk",
                    content="",
                    role="assistant",
                    model=model,
                    docs=source_documents,
                )
                yield ret.model_dump_json()

                async for token in callback.aiter():
                    ret = OpenAIChatOutput(
                        id=f"chat{uuid.uuid4()}",
                        object="chat.completion.chunk",
                        content=token,
                        role="assistant",
                        model=model,
                    )
                    yield ret.model_dump_json()
            else:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                ret = OpenAIChatOutput(
                    id=f"chat{uuid.uuid4()}",
                    object="chat.completion",
                    content=answer,
                    role="assistant",
                    model=model,
                )
                yield ret.model_dump_json()
            await task

        except Exception as e:
            yield {"data": json.dumps({"error": str(e)})}
            return
    if stream:
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:
        return await knowledge_base_chat_iterator().__anext__()

@app.post("/kb_chat", summary="知识库对话")
async def kb_chat_endpoint(query: str = Body(..., description="用户输入", example=["你好"]),
                          mode: Literal["local_kb"] = Body("local_kb", description="知识来源"),
                          top_k: int = Body(3, description="匹配向量数字"),
                          score_threshold: float = Body(
                              2.0,
                              description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                              ge=0,
                              le=2,
                          ),
                          kb_name: str = Body("",
                                              description="mode=local_kb时为知识库名称；temp_kb时为临时知识库ID，search_engine时为搜索引擎名称",
                                              examples=["samples"]),

                          stream: bool = Body(True, description="流式输出"),
                          model: str = Body("qwen:1.8b", description="LLM 模型名称。"),
                          temperature: float = Body(0.7, description="LLM 采样温度", ge=0.0,
                                                    le=2.0),
                          max_tokens: Optional[int] = Body(
                              None,
                              description="限制LLM生成Token数量，默认None代表模型最大值"
                          ),
                          prompt_name: str = Body(
                              "default",
                              description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                          ),
                          return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM")):
    # 调用 kb_chat 函数
    return await kb_chat(query=query, mode=mode, top_k=top_k, score_threshold=score_threshold,
                   kb_name=kb_name, stream=stream, model=model, temperature=temperature,
                   max_tokens=max_tokens, prompt_name=prompt_name, return_direct=return_direct)





if __name__ == "__main__":
    import uvicorn
    # 绑定到 localhost 只允许本地访问
    uvicorn.run(app, host="localhost", port=8000)
