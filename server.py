import os

import psycopg
from langchain_core.documents import Document
from langchain_ollama import OllamaLLM, ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages

from db.repository.knowledge_file_repository import get_file_detail
from knowledge_base.utils import get_file_path, KnowledgeFile, files2docs_in_thread, get_kb_path

os.environ["OTEL_SDK_DISABLED"] = "true"
# server.py
import pprint
import uuid
from datetime import datetime

from fastapi import FastAPI, Request, Body, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from langchain_classic.callbacks import AsyncIteratorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncIterable, Literal, List, TypedDict, Annotated
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

from utils import format_reference, get_ChatOpenAI, wrap_done, get_prompt_template, History, run_in_thread_pool
# 在导入语句之后，FastAPI应用创建之前添加
from db.base import Base, engine
from utils import build_logger
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

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
            logger.info("开始执行")
            docs = kb.search_docs(query, top_k, score_threshold)
            # data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
            data = [DocumentWithVSId(**{"id": x.metadata.get("id"), **x.dict()}) for x in docs]
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
    return [x.dict() for x in data]


@app.post("/api/kb_chat", summary="知识库对话")
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
                  prompt_name: str = Body(
                      "default",
                      description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                  ),
                  return_direct: bool = Body(False, description="直接返回检索结果，不送入 LLM"),
                  request: Request = None,
                  ):
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            nonlocal prompt_name
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

            llm = ChatOllama(model="qwen:1.8b", temperature=0.7, callbacks=callbacks)

            context = "\n\n".join([doc["page_content"] for doc in docs])
            logger.info(f"这是chat1的context{context}")

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
            logger.exception(e)
            yield {"data": json.dumps({"error": str(e)})}
            return

    if stream:
        return EventSourceResponse(knowledge_base_chat_iterator())
    else:
        return await knowledge_base_chat_iterator().__anext__()


from psycopg_pool import ConnectionPool
from contextlib import asynccontextmanager
import os

# 全局连接池
pool = None


@app.post("/api/kb_chat2", summary="知识库对话")
async def kb_chat(query: str = Body(..., description="用户输入", example=["你好"]),
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
                  prompt_name: str = Body(
                      "default",
                      description="使用的prompt模板名称(在prompt_settings.yaml中配置)"
                  ),
                  model: str = Body("qwen:1.8b", description="LLM 模型名称。"),

                  ):
    async def knowledge_base_chat_iterator() -> AsyncIterable[str]:
        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            async with aiosqlite.connect("checkpoints.sqlite") as conn:
                checkpointer = AsyncSqliteSaver(conn)
                # checkpointer.setup()

                class KBChatState(TypedDict):
                    messages: Annotated[list[BaseMessage], add_messages]
                    context: str
                    sources: str
                    question: str

                async def retrieve_documents(state: KBChatState) -> KBChatState:
                    last_message = state["messages"][-1].content
                    docs = search_docs(
                        query=query,
                        knowledge_base_name=kb_name,
                        top_k=top_k,
                        score_threshold=score_threshold,
                        file_name="",
                        metadata={}
                    )
                    source_documents = format_reference(kb_name, docs, "")
                    context = "\n\n".join([doc.get("page_content", "") for doc in docs])

                    return {
                        "context": context,
                        "sources": source_documents,
                        "question": last_message,
                    }

                async def generate_response(state: KBChatState) -> KBChatState:
                    if not state["context"] or state["context"].strip() == "":
                        response = "根据提供的资料无法回答您的问题。知识库中不包含相关信息。"
                        return {"messages": [AIMessage(content=response)]}

                    prompt_template = get_prompt_template("rag", prompt_name)

                    # 限制历史消息数量，只保留最近的4条消息（2轮对话）
                    all_messages = state["messages"]
                    if len(all_messages) > 4:
                        recent_messages = all_messages[-4:]
                        logger.info(f"历史消息过多，仅保留最近4条消息")
                    else:
                        recent_messages = all_messages

                    # 构建历史消息列表，正确处理各种消息类型
                    history_messages = []
                    for msg in recent_messages:
                        if isinstance(msg, HumanMessage):
                            history_messages.append(History(role="user", content=msg.content).to_msg_template())
                        elif isinstance(msg, AIMessage):
                            history_messages.append(History(role="assistant", content=msg.content).to_msg_template())

                    # 添加当前问题的模板
                    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
                    chat_prompt = ChatPromptTemplate.from_messages(history_messages + [input_msg])
                    logger.info(f"总共使用 {len(history_messages)} 条历史消息")

                    llm = ChatOllama(
                        model="qwen:1.8b",
                        temperature=0.7,
                    )
                    chain = chat_prompt | llm | StrOutputParser()

                    try:
                        response = await chain.ainvoke({
                            "context": state["context"],
                            "sources": state["sources"] if state["sources"] else "未知来源",
                            "question": state["question"],
                        })
                        logger.info(f"模型响应长度: {len(response)} 字符")

                        if not response:
                            response = "无法生成答案，请稍后重试。"
                        return {"messages": [AIMessage(content=response)]}

                    except Exception as e:
                        logger.error(f"LLM调用失败: {str(e)}")
                        return {"messages": [AIMessage(content=f"处理过程中出错: {str(e)}")]}

                workflow = StateGraph(KBChatState)
                workflow.add_node("retrieve", retrieve_documents)
                workflow.add_node("generate", generate_response)
                workflow.add_edge(START, "retrieve")
                workflow.add_edge("retrieve", "generate")
                workflow.add_edge("generate", END)

                kb_app = workflow.compile(checkpointer=checkpointer)

                config = {"configurable": {"thread_id": "default_thread"}}

                # 关键：从数据库读取历史消息
                state_snapshot = await checkpointer.aget(config)
                # logger.info(f"stage_snapshot:{state_snapshot.get('channel_values', {})}")
                history_messages = state_snapshot.get('channel_values', {}).get('messages',
                                                                                []) if state_snapshot else []
                # 新消息追加到历史后面
                all_messages = history_messages + [HumanMessage(content=query)]

                inputs = {"messages": all_messages}

                async for event in kb_app.astream(inputs, stream_mode="values", config=config):
                    if isinstance(event, dict) and "messages" in event:
                        messages = event["messages"]
                        if messages:
                            latest_message = messages[-1]
                            if isinstance(latest_message, AIMessage):
                                content = latest_message.content
                                logger.info(f"最终输出: {content[:100]}")

                                if not isinstance(content, str):
                                    content = str(content)

                                ret = OpenAIChatOutput(
                                    id=f"chat{uuid.uuid4()}",
                                    object="chat.completion.chunk",
                                    content=content,
                                    role="assistant",
                                    model=model,
                                )
                                yield ret.model_dump_json()

        except Exception as e:
            logger.exception(e)
            yield json.dumps({"error": str(e)})
            return
    return EventSourceResponse(knowledge_base_chat_iterator())


def _save_files_in_thread(
        files: List[UploadFile], knowledge_base_name: str, override: bool
):
    """
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        """
        保存单个文件。
        """
        try:
            filename = file.filename
            file_path = get_file_path(
                knowledge_base_name=knowledge_base_name, doc_name=filename
            )
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容
            if (
                    os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            return dict(code=500, msg=msg, data=data)

    params = [
        {"file": file, "knowledge_base_name": knowledge_base_name, "override": override}
        for file in files
    ]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


def update_docs(
        knowledge_base_name: str = Body(
            ..., description="知识库名称", examples=["samples"]
        ),
        file_names: List[str] = Body(
            ..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]
        ),
        chunk_size: int = Body(750, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(150, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(False, description="是否开启中文标题加强"),
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        docs: str = Body("", description="自定义的docs，需要转为json字符串"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
):
    """
    更新知识库文档
    """
    # if not validate_kb_name(knowledge_base_name):
    #     return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        # return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
        return {"status": 'Fail', "message": f"未找到知识库 {knowledge_base_name}", "data": None}

    failed_files = {}
    kb_files = []
    docs = json.loads(docs) if docs else {}

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            try:
                kb_files.append(
                    KnowledgeFile(
                        filename=file_name, knowledge_base_name=knowledge_base_name
                    )
                )
            except Exception as e:
                msg = f"加载文档 {file_name} 时出错：{e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                failed_files[file_name] = msg

    # 从文件生成docs，并进行向量化。
    # 这里利用了KnowledgeFile的缓存功能，在多线程中加载Document，然后传给KnowledgeFile
    for status, result in files2docs_in_thread(
            kb_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
    ):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return {"status": 'Fail', "message": "成功", "data": {"failed_files": failed_files}}


@app.post("/upload_docs", summary="上传文件到知识库并进行向量化")
def upload_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        knowledge_base_name: str = Form(
            ..., description="知识库名称", examples=["samples"]
        ),
        override: bool = Form(False, description="覆盖已有文件"),
        to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
        chunk_size: int = Form(750, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(150, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(False, description="是否开启中文标题加强"),
        docs: str = Form("", description="自定义的docs，需要转为json字符串"),
        not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
):
    """
    API接口：上传文件，并/或向量化
    """
    # if not validate_kb_name(knowledge_base_name):
    #     return {"status": 'Fail', "message": "Don not attack me", "data": None}

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return {"status": 'Fail', "message": "未找到知识库", "data": None}

    docs = json.loads(docs) if docs else {}
    failed_files = {}
    file_names = list(docs.keys())

    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(
            files, knowledge_base_name=knowledge_base_name, override=override
    ):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        # logger.info(f"更新文档结果：{result['data']['failed_files']}")
        failed_files.update(result['data']['failed_files'])
        # failed_files.update(result.data["failed_files"])

        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return {"status": 'success', "message": "成功", "data": None}


if __name__ == "__main__":
    import uvicorn

    # 绑定到 localhost 只允许本地访问
    get_kb_path("samples")
    uvicorn.run(app, host="localhost", port=8000)
