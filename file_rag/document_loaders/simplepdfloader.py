from typing import List
import os
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_core.documents import Document
from utils import build_logger

logger = build_logger(__name__)


class SimplePDFLoader(UnstructuredFileLoader):
    """
    简单的PDF文档加载器，只提取文本内容，不进行OCR处理
    """
    def _get_elements(self) -> List:
        """
        从PDF文件中提取元素
        """
        try:
            # 尝试使用PyPDFLoader
            from langchain_community.document_loaders import PyPDFLoader
            pypdf_loader = PyPDFLoader(file_path=self.file_path)
            pages = pypdf_loader.load()
            
            # 将页面内容合并
            full_text = "\n".join([page.page_content for page in pages if page.page_content])
            
        except Exception as e:
            logger.warning(f"使用PyPDFLoader加载失败 {self.file_path}: {e}，尝试使用PDFPlumberLoader")
            try:
                # 如果PyPDFLoader失败，尝试使用PDFPlumberLoader
                from langchain_community.document_loaders import PDFPlumberLoader
                pdfplumber_loader = PDFPlumberLoader(file_path=self.file_path)
                pages = pdfplumber_loader.load()
                
                # 将页面内容合并
                full_text = "\n".join([page.page_content for page in pages if page.page_content])
                
            except Exception as e2:
                logger.error(f"使用PDFPlumberLoader也失败 {self.file_path}: {e2}")
                raise e  # 抛出第一个错误

        # 使用unstructured库的partition_text函数来处理文本
        from unstructured.partition.text import partition_text
        return partition_text(text=full_text, **self.unstructured_kwargs)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        loader = SimplePDFLoader(file_path=sys.argv[1])
        docs = loader.load()
        print(f"成功加载 {len(docs)} 个文档片段")
        for i, doc in enumerate(docs[:3]):  # 只打印前3个片段
            print(f"文档片段 {i+1}: {doc.page_content[:200]}...")  # 只显示前200个字符
    else:
        print("请提供PDF文件路径作为参数")