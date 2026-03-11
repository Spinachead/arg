from __future__ import annotations

from typing import Dict, List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from file_rag.retrievers.base import BaseRetrieverService
from utils import build_logger

logger = build_logger()


class BM25IndexCache:
    """BM25 索引缓存管理器 - 避免每次查询都重建索引"""
    
    _cache: Dict[str, BM25Retriever] = {}
    _docs_hash: Dict[str, int] = {}
    _docs_count: Dict[str, int] = {}
    
    @classmethod
    def get_index(
        cls, 
        cache_key: str, 
        docs: List[Document],
        preprocess_func=None
    ) -> Optional[BM25Retriever]:
        """
        获取缓存的 BM25 索引，如果文档变化则重建
        
        Args:
            cache_key: 缓存键（如知识库名称）
            docs: 文档列表
            preprocess_func: 分词函数
        
        Returns:
            BM25Retriever 实例
        """
        current_hash = hash(tuple(doc.page_content[:100] for doc in docs[:1000]))
        
        # 检查缓存是否有效
        if cache_key in cls._cache:
            cached_hash = cls._docs_hash.get(cache_key)
            if cached_hash == current_hash and len(docs) == cls._docs_count.get(cache_key, 0):
                logger.info(f"Using cached BM25 index for '{cache_key}' ({len(docs)} docs)")
                return cls._cache[cache_key]
        
        # 重建索引
        logger.info(f"Building BM25 index for '{cache_key}' ({len(docs)} docs)...")
        import jieba
        
        bm25_retriever = BM25Retriever.from_documents(
            docs,
            preprocess_func=preprocess_func or jieba.lcut_for_search,
        )
        
        # 更新缓存
        cls._cache[cache_key] = bm25_retriever
        cls._docs_hash[cache_key] = current_hash
        cls._docs_count[cache_key] = len(docs)
        
        logger.info(f"BM25 index built successfully for '{cache_key}'")
        return bm25_retriever
    
    @classmethod
    def invalidate(cls, cache_key: str):
        """使指定缓存失效"""
        cls._cache.pop(cache_key, None)
        cls._docs_hash.pop(cache_key, None)
        cls._docs_count.pop(cache_key, None)
        logger.info(f"BM25 cache invalidated for '{cache_key}'")
    
    @classmethod
    def clear_all(cls):
        """清空所有缓存"""
        cls._cache.clear()
        cls._docs_hash.clear()
        cls._docs_count.clear()
        logger.info("All BM25 caches cleared")


class EnsembleRetrieverService(BaseRetrieverService):
    def do_init(
        self,
        retriever: BaseRetriever = None,
        top_k: int = 5,
        cache_key: str = "default",
    ):
        self.vs = None
        self.top_k = top_k
        self.retriever = retriever
        self.cache_key = cache_key

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
                        ids = raw.get("ids", [None] * len(raw["documents"]))
                        metadatas = raw.get("metadatas", [{}] * len(raw["documents"]))
                        for i, meta in enumerate(metadatas):
                            if meta is None:
                                metadatas[i] = {}
                            if ids[i]:
                                metadatas[i]["id"] = ids[i]
                        
                        docs = [
                            Document(page_content=doc, metadata=meta)
                            for doc, meta in zip(raw["documents"], metadatas)
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
            
            # 创建关键词检索器（BM25）- 使用缓存避免重复构建
            import jieba
            
            # 从向量库获取知识库名称作为缓存键
            cache_key = getattr(vectorstore, '_collection', None)
            if cache_key:
                cache_key = getattr(cache_key, 'name', 'default')
            else:
                cache_key = 'default'
            
            bm25_retriever = BM25IndexCache.get_index(
                cache_key=cache_key,
                docs=docs,
                preprocess_func=jieba.lcut_for_search,
            )
            
            if bm25_retriever is None:
                logger.error("Failed to build BM25 index")
                return None
            
            bm25_retriever.k = top_k
            
            # 创建混合检索器
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, semantic_retriever], weights=[0.5, 0.5]
            )
            
        except Exception as e:
            logger.exception(f"Error creating ensemble retriever: {e}")
            return None
        return EnsembleRetrieverService(
            retriever=ensemble_retriever, 
            top_k=top_k,
            cache_key=cache_key
        )

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