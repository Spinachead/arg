# server.py
from fastapi import FastAPI
from rag_chain import create_rag_chain
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware

# 确保 Ollama 正在运行！
app = FastAPI(
    title="RAG API Service",
    version="1.0",
    description="基于 Qwen + 本地文档的问答 API"
)

#添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该指定具体的域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# 创建 RAG Chain（首次会构建向量库）
rag_chain = create_rag_chain()

# 挂载到 /rag 路径
add_routes(
    app,
    rag_chain,
    path="/api"
)

@app.get("/")
async def root():
    return {"message": "RAG Server is running. Visit /docs for API documentation or /rag/playground for playground."}

@app.post("/api/chat-process")
async def chat(request: dict):
    question = request.get("input", "")
    if not question:
        return {"status": 'Error', "data": None, "message": "Question input is required"}
    
    try:
        response = rag_chain.invoke(question)
        return {
            "status": 'Success', 
            "data": {
                'id': "chat-1", 
                'role': "assistant", 
                'text': response, 
                'dateTime': "1111111"
            }, 
            "message": "Success"
        }
    except Exception as e:
        return {"status": 'Error', "data": None, "message": str(e)}

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