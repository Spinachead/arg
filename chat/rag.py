import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.tools import create_retriever_tool
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def setup_rag_tools():
    try:
        # Initialize Embeddings using OpenAI
        embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

        # Load Data from LangChain Blog
        # Using WebBaseLoader to scrape the blog post
        loader = WebBaseLoader(
            web_paths=("https://blog.langchain.com/langchain-langgraph-1dot0/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        # Vector Store using langchain-qdrant
        client = QdrantClient(":memory:")

        # Create collection if it doesn't exist (it won't in memory)
        # We need to know the vector size. OpenAI text-embedding-3-small is 1536.
        client.create_collection(
            collection_name="langchain_blog",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name="langchain_blog",
            embedding=embedding_model,
        )

        # Add documents
        vectorstore.add_documents(splits)

        retriever = vectorstore.as_retriever()

        retriever_tool = create_retriever_tool(
            retriever,
            "search_langchain_updates",
            "Search for information about LangChain 1.0 and LangGraph 1.0 releases."
        )

        return [retriever_tool]
    except Exception as e:
        # 如果加载失败，返回空工具列表或基本工具
        print(f"Error setting up RAG tools: {e}")
        return []