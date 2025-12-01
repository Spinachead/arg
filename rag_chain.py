# rag_chain_with_memory.py
import torch.cuda
from pathlib import Path
from typing import TypedDict, Annotated

from langchain.agents import create_agent
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages


# 定义状态结构（包含对话历史）
class RagState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # 对话历史
    context: str  # RAG检索到的文档
    sources: str  # 文档来源
    question: str  # 当前问题


def create_rag_chain():
    # 初始化嵌入和向量库
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": device}
    )
    persist_dir = str((Path(__file__).parent / "chroma_db").resolve())
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model="qwen:1.8b", temperature=0.7)
    checkpointer = InMemorySaver()
    agent = create_agent(
        model=llm,
        checkpointer=checkpointer,
    )

    # 步骤1：检索相关文档
    def retrieve_docs(state: RagState) -> RagState:
        """从最后一条用户消息检索相关文档"""
        # 获取最后一条用户消息作为查询
        last_message = state["messages"][-1].content
        docs = retriever.invoke(last_message)

        # 格式化文档
        sources = ", ".join(set(doc.metadata["source"] for doc in docs))
        context = "\n\n".join(doc.page_content for doc in docs)

        return {
            "context": context,
            "sources": sources,
            "question": last_message,
        }

    # 步骤2：使用RAG提示调用LLM
    def rag_response(state: RagState) -> RagState:
        """基于检索内容和对话历史生成回复"""
        template = """你是一个专业助手，请严格根据以下上下文回答问题。
如果上下文没有相关信息，请回答"根据提供的资料无法回答"。

上下文（来自 {sources}）：
{context}

对话历史：
{history}

问题：{question}
"""

        # 构建对话历史
        history = "\n".join([
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in state["messages"][:-1]  # 排除最后一条（当前问题）
        ])
        print(history)

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | agent | StrOutputParser()

        response = chain.invoke({
            "context": state["context"],
            "sources": state["sources"],
            "question": state["question"],
            "history": history if history else "无"
        })

        # 将LLM响应添加到消息历史
        return {
            "messages": [AIMessage(content=response)],
            "context": state["context"],
            "sources": state["sources"],
            "question": state["question"]
        }

    return agent


# 使用示例
if __name__ == "__main__":
    agent = create_rag_chain()
    # 配置线程ID（表示同一个对话会话）
    config = RunnableConfig({"configurable": {"thread_id": "user_123"}})

    # 第一条消息
    input1 = {
        "messages": [HumanMessage(content="我的名字叫Bob？")]
    }
    result1 = agent.invoke(input1, config=config)
    print("问题1:", input1["messages"][0].content)
    print("回复1:", result1["messages"][-1].content)
    print()

    # 第二条消息（记忆仍然存在）
    input2 = {
        "messages": [HumanMessage(content="我今年12岁了？")]
    }
    result2 = agent.invoke(input2, config=config)
    print("问题2:", input2["messages"][0].content)
    print("回复2:", result2["messages"][-1].content)


    final_response = agent.invoke({"messages": [HumanMessage(content="你知道我的名字吗,只说出名字就好")]}, config)

    final_response["messages"][-1].pretty_print()