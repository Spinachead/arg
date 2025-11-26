# build_vectorstore.py
from document_loader import load_documents_from_directory
from chunking import smart_split_documents
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


def build_or_update_vectorstore(doc_dir: str, persist_dir: str = "./chroma_db"):
    """构建或增量更新向量库"""
    # 1. 加载所有文档
    docs = load_documents_from_directory(doc_dir)

    # 2. 智能分块
    chunks = smart_split_documents(docs)

    # 3. 初始化嵌入模型
    embedding = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"}
    )

    # 4. 创建或更新向量库
    if os.path.exists(persist_dir):
        print("🔄 更新现有向量库...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding
        )
        # 删除旧数据（简单方案：重建；生产环境可用 delete(ids)）
        # 这里为简化，直接重建
        import shutil
        shutil.rmtree(persist_dir)

    print("🆕 创建新向量库...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )

    print(f"✅ 向量库构建完成！共 {len(chunks)} 个片段")
    return vectorstore