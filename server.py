# server.py
import pprint

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, AsyncIterable
import asyncio
import json
from rag_chain import create_rag_graph
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage

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

# @app.post("/api/chat-process")
# async def chat_process_stream(request: ChatProcessRequest):
#     async def generate_response() -> AsyncIterable[str]:
#         try:
#             if not request.prompt:
#                 error_data = {
#                     'status': 'Error',
#                     'data': None,
#                     'message': 'Question input is required'
#                 }
#                 yield f"{json.dumps(error_data)}\n"
#                 return
#
#             # 获取会话ID，用于记忆功能
#             session_id = request.options.get("sessionId", "default_session") if request.options else "default_session"
#             config = {"configurable": {"thread_id": session_id}}
#
#             # 使用带记忆功能的 RAG 链处理请求
#             input_data = {
#                 "messages": [HumanMessage(content=request.prompt)]
#             }
#             response = rag_chain.invoke(input_data, config=config)
#             result = response["messages"][-1].content
#
#             # 流式输出，按字符逐个输出
#             for i, char in enumerate(result):
#                 chat_data = {
#                     "id": "chat-1",
#                     "role": "assistant",
#                     "text": result[:i + 1],
#                     "dateTime": "1111111"
#                 }
#                 yield f"{json.dumps(chat_data)}\n"
#                 await asyncio.sleep(0.01)  # 控制输出速度
#
#         except Exception as e:
#             error_data = {
#                 'status': 'Error',
#                 'data': None,
#                 'message': str(e)
#             }
#             yield f"{json.dumps(error_data)}\n"
#
#     return StreamingResponse(generate_response(), media_type="text/event-stream")


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
            print(input_data)

            # 使用 stream 方法进行真正的流式输出
            accumulated_content = ""
            full_response = ""
            
            # 处理流式响应
            for chunk in rag_chain.stream(input_data, config=config):
                print("chunk", chunk)
                # 检查chunk是否包含消息
                if isinstance(chunk, dict) and "messages" in chunk:
                    if chunk["messages"]:
                        latest_message = chunk["messages"][-1]
                        if hasattr(latest_message, 'content'):
                            new_content = latest_message.content
                            delta = new_content[len(accumulated_content):]
                            if delta:
                                accumulated_content = new_content
                                chat_data = {
                                    "id": "chat-1",
                                    "role": "assistant",
                                    "text": accumulated_content,
                                    "dateTime": "1111111"
                                }
                                yield f"{json.dumps(chat_data)}\n"
                elif isinstance(chunk, dict):
                    # 处理嵌套结构的chunk（如 {'rag_response': {'messages': [...]}}）
                    for key, value in chunk.items():
                        if isinstance(value, dict) and "messages" in value:
                            if value["messages"]:
                                latest_message = value["messages"][-1]
                                if hasattr(latest_message, 'content'):
                                    new_content = latest_message.content
                                    delta = new_content[len(full_response):]
                                    if delta:
                                        full_response = new_content
                                        chat_data = {
                                            "id": "chat-1",
                                            "role": "assistant",
                                            "text": full_response,
                                            "dateTime": "1111111"
                                        }
                                        yield f"{json.dumps(chat_data)}\n"
                        elif hasattr(value, 'content'):
                            print("执行到这里")
                            new_content = value.content
                            delta = new_content[len(full_response):]
                            if delta:
                                full_response = new_content
                                chat_data = {
                                    "id": "chat-1",
                                    "role": "assistant",
                                    "text": full_response,
                                    "dateTime": "1111111"
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
    return {"status": 'Success', "data":{'apiModel': 'Mock-API', 'socksProxy': 'false', 'httpsProxy': 'false'}, "message": "Success"}

@app.post("/api/session")
async def session():
    return {"status": 'Success', "data":{'auth': 'true', 'model': 'Mock-API'}, "message": "null"}

@app.post("/api/verify")
async def verify():
    return {"status": 'Success', "data": None, "message": "Verify successfully"}


if __name__ == "__main__":
    import uvicorn
    # 绑定到 localhost 只允许本地访问
    uvicorn.run(app, host="localhost", port=8000)