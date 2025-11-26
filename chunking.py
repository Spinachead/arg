# chunking.py
from langchain_text_splitters import RecursiveCharacterTextSplitter


def smart_split_documents(documents):
    """根据文档类型智能分块"""
    final_chunks = []

    for doc in documents:
        source = doc.metadata.get("source", "")

        if source.endswith('.md') or source.endswith('.xmind'):
            # Markdown 和脑图：按标题分块（保留层级）
            splitter = RecursiveCharacterTextSplitter(
                separators=["\n## ", "\n### ", "\n#### ", "\n---", "\n\n"],
                chunk_size=600,
                chunk_overlap=80,
                keep_separator=True
            )
        elif source.endswith('.docx'):
            # Word：适当增大块大小
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )
        else:
            # PDF 默认
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )

        chunks = splitter.split_documents([doc])
        final_chunks.extend(chunks)

    return final_chunks