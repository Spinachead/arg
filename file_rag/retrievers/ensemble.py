from __future__ import annotations

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from file_rag.retrievers.base import BaseRetrieverService
from utils import build_logger
logger = build_logger()


class EnsembleRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever

    @staticmethod
    def from_vectorstore(
        vectorstore: VectorStore,
        top_k: int,
        score_threshold: int | float,
    ):
        try:
            # 根据向量库类型获取所有文档
            docs = []
            
            # 检查是否是 Chroma 向量库
            if hasattr(vectorstore, '_collection'):
                # Chroma 向量库
                from langchain_core.documents import Document
                logger.info("Detected Chroma vectorstore")
                try:
                    raw = vectorstore._collection.get(include=["documents", "metadatas"])
                    if raw and raw.get("documents"):
                        docs = [
                            Document(page_content=doc, metadata=meta or {})
                            for doc, meta in zip(raw["documents"], raw.get("metadatas", [{}] * len(raw["documents"])))
                        ]
                        logger.info(f"Found {len(docs)} documents in Chroma vectorstore")
                    else:
                        logger.warning("No documents found in Chroma vectorstore")
                        return None
                except Exception as e:
                    logger.error(f"Error getting documents from Chroma: {e}")
                    return None
                    
            # 检查是否是 FAISS 等有 docstore 的向量库
            elif hasattr(vectorstore, 'docstore') and vectorstore.docstore:
                # FAISS 等向量库
                logger.info("Detected FAISS or similar vectorstore")
                docs = list(vectorstore.docstore._dict.values())
                if not docs:
                    logger.warning("No documents found in vectorstore")
                    return None
                logger.info(f"Found {len(docs)} documents in vectorstore")
            else:
                logger.warning("Unsupported vectorstore type or empty vectorstore")
                return None
            
            # 创建语义检索器（向量检索）
            semantic_retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": top_k},
            )
            
            # 创建关键词检索器（BM25）
            # TODO: 换个不用torch的实现方式
            # from cutword.cutword import Cutter
            import jieba

            # cutter = Cutter()
            bm25_retriever = BM25Retriever.from_documents(
                docs,
                preprocess_func=jieba.lcut_for_search,
            )
            bm25_retriever.k = top_k
            
            # 创建混合检索器
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5]
            )
            
        except Exception as e:
            logger.exception(f"Error creating ensemble retriever: {e}")
            return None
        return EnsembleRetrieverService(retriever=ensemble_retriever, top_k=top_k)

    def get_relevant_documents(self, query: str):
        if self.retriever is None:
            logger.warning("Retriever is None")
            return []
        try:
            docs = self.retriever.invoke(query)
            return docs[: self.top_k]
        except Exception as e:
            logger.exception(f"Error getting relevant documents: {e}")
            return []