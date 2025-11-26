# server.py
from fastapi import FastAPI
from rag_chain import create_rag_chain
from langserve import add_routes

# 确保 Ollama 正在运行！
app = FastAPI(
    title="RAG API Service",
    version="1.0",
    description="基于 Qwen + 本地文档的问答 API"
)

# 创建 RAG Chain（首次会构建向量库）
rag_chain = create_rag_chain()

# 挂载到 /rag 路径
add_routes(
    app,
    rag_chain,
    path="/rag"
)

@app.get("/")
async def root():
    return {"message": "RAG Server is running. Visit /docs for API documentation or /rag/playground for playground."}

if __name__ == "__main__":
    import uvicorn
    # 绑定到 localhost 只允许本地访问
    uvicorn.run(app, host="localhost", port=8000)