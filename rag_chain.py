# rag_chain.py（简化版）
import torch.cuda
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

def create_rag_chain():
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

    def prepare_inputs(question: str):
        """兼容 LangServe 默认的字符串输入。"""
        if isinstance(question, dict):
            # 允许 /rag/playground 这类场景传递 {"input": "..."} 结构
            question = question.get("input", "")
        else:
            question = str(question)
            
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