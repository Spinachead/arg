from typing import Literal
from pathlib import Path
import torch.cuda

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict, Annotated
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma


# 定义状态结构
class RagState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: str
    sources: str
    question: str


def create_rag_graph():
    # 初始化
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # embedding = HuggingFaceEmbeddings(
    #     model_name="BAAI/bge-small-zh-v1.5",
    #     model_kwargs={"device": device}
    # )
    embedding = OllamaEmbeddings(
        model="bge-small-zh-v1.5",
        base_url="http://localhost:11434"
    )
    persist_dir = str((Path(__file__).parent / "chroma_db").resolve())
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model="qwen:1.8b", temperature=0.7)

    # 节点1：检索相关文档
    def retrieve_docs(state: RagState) -> RagState:
        """从最后一条用户消息检索相关文档"""
        last_message = state["messages"][-1].content
        docs = retriever.invoke(last_message)

        # 格式化文档
        sources = ", ".join(set(doc.metadata.get("source", "未知") for doc in docs))
        context = "\n\n".join(doc.page_content for doc in docs)

        return {
            "context": context,
            "sources": sources,
            "question": last_message,
        }

    # 节点2：生成回复
    def rag_response(state: RagState) -> RagState:
        """基于检索内容生成回复"""
        template = """你是一个专业助手，请严格根据以下上下文回答问题。
如果上下文没有相关信息，请回答"根据提供的资料无法回答"。

上下文（来自 {sources}）：
{context}

对话历史：
{history}

问题：{question}
"""

        # 构建对话历史（排除当前问题）
        history = "\n".join([
            f"{msg.__class__.__name__}: {msg.content}"
            for msg in state["messages"][:-1]
        ])

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": state["context"],
            "sources": state["sources"],
            "question": state["question"],
            "history": history if history else "无"
        })

        # 返回状态更新：新增 AI 消息
        return {
            "messages": [AIMessage(content=response)],
        }

    # 构建图
    builder = StateGraph(RagState)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("rag_response", rag_response)

    # 定义流程
    builder.add_edge(START, "retrieve_docs")
    builder.add_edge("retrieve_docs", "rag_response")
    builder.add_edge("rag_response", END)

    # 编译图并配置检查点
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


# 使用示例
if __name__ == "__main__":
    graph = create_rag_graph()
    config = {"configurable": {"thread_id": "user_123"}}

    # 第一条消息
    result1 = graph.invoke(
        {"messages": [HumanMessage(content="我的名字叫姜波")]},
        config=config
    )
    print("问题1:", result1["messages"][-2].content)
    print("回复1:", result1["messages"][-1].content)
    print()

    # 第二条消息 - 自动包含历史
    result2 = graph.invoke(
        {"messages": [HumanMessage(content="我今年12岁了")]},
        config=config
    )
    print("问题2:", result2["messages"][-2].content)
    print("回复2:", result2["messages"][-1].content)
    print()

    # 第三条消息
    result3 = graph.invoke(
        {"messages": [HumanMessage(content="说出我今年多大了叫什么名字，只说我的名字和年龄不要说其他的")]},
        config=config
    )
    print("问题3:", result3["messages"][-2].content)
    result3["messages"][-1].pretty_print()