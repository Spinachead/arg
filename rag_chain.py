# rag_chain.py（简化版）
import torch.cuda
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": device}
    )
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embedding
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}  # 返回最相关的 4 个片段
    )

    llm = ChatOllama(model="qwen:1.8b", temperature=0.7)

    template = """你是一个专业助手，请严格根据以下上下文回答问题。
如果上下文没有相关信息，请回答“根据提供的资料无法回答”。

上下文（来自 {sources}）：
{context}

问题：{question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        sources = ", ".join(set(doc.metadata["source"] for doc in docs))
        context = "\n\n".join(doc.page_content for doc in docs)
        return context, sources

    def prepare_inputs(input_data):
        # 从LangServe的标准输入格式中提取question
        if isinstance(input_data, dict):
            question = input_data.get("input", "")
        else:
            question = str(input_data)
            
        docs = retriever.invoke(question)
        context, sources = format_docs(docs)
        return {
            "context": context,
            "sources": sources,
            "question": question
        }

    rag_chain = (
        prepare_inputs
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain