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
            # 检查向量存储是否为空
            if not hasattr(vectorstore, 'docstore') or not vectorstore.docstore:
                logger.warning("Vectorstore is empty or doesn't have docstore")
                return None
            
            docs = list(vectorstore.docstore._dict.values())
            if not docs:
                logger.warning("No documents found in vectorstore")
                return None
                
            logger.info(f"Found {len(docs)} documents in vectorstore")
            
            faiss_retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": score_threshold, "k": top_k},
            )
            # TODO: 换个不用torch的实现方式
            # from cutword.cutword import Cutter
            import jieba

            # cutter = Cutter()
            bm25_retriever = BM25Retriever.from_documents(
                docs,
                preprocess_func=jieba.lcut_for_search,
            )
            bm25_retriever.k = top_k
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5]
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