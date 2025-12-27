from __future__ import annotations


from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from file_rag.retrievers.base import BaseRetrieverService

from utils import build_logger
logger = build_logger()


class VectorstoreRetrieverService(BaseRetrieverService):
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
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": top_k},
        )
        # retriever = vectorstore.as_retriever(
        #     search_type="similarity",  # 改为简单相似度搜索
        #     search_kwargs={"k": top_k},
        # )
        return VectorstoreRetrieverService(retriever=retriever, top_k=top_k)


    def get_relevant_documents(self, query: str):
        if self.retriever is None:
            logger.warning("Retriever is None")
            return []

        try:
            # 先检查是否能调用检索器
            docs = self.retriever.invoke(query)
            # 如果返回空列表，打印调试信息
            if not docs:
                logger.warning(f"No documents found for query: {query}")
                logger.info(f"Retriever type: {type(self.retriever)}")
                logger.info(f"Retriever search_kwargs: {getattr(self.retriever, 'search_kwargs', 'N/A')}")

            return docs[: self.top_k]
        except Exception as e:
            logger.exception(f"Error getting relevant documents: {e}")
            return []
