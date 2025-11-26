# document_loader.py
import os
from pathlib import Path
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document


# 脑图解析（XMind）
def load_xmind(file_path: str) -> list[Document]:
    """将 XMind 脑图转为层级文本"""
    try:
        from xmindparser import xmind_to_dict
        content = xmind_to_dict(file_path)

        def parse_topic(topic, level=0):
            title = topic.get('title', '')
            indent = "  " * level
            text = f"{indent}- {title}\n"

            children = topic.get('children', {}).get('attached', [])
            for child in children:
                text += parse_topic(child, level + 1)
            return text

        full_text = ""
        for sheet in content:
            root_topic = sheet.get('topic', {})
            full_text += parse_topic(root_topic)

        return [Document(page_content=full_text, metadata={"source": file_path})]

    except Exception as e:
        print(f"⚠️  脑图解析失败 {file_path}: {e}")
        return []


def load_documents_from_directory(dir_path: str) -> list[Document]:
    """加载目录下所有支持的文档"""
    all_docs = []
    supported_ext = {'.pdf', '.docx', '.md', '.xmind'}

    for file_path in Path(dir_path).rglob('*'):
        if file_path.suffix.lower() not in supported_ext:
            continue

        print(f"📄 正在加载: {file_path}")
        try:
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                docs = loader.load()

            elif file_path.suffix.lower() == '.docx':
                loader = UnstructuredWordDocumentLoader(
                    str(file_path),
                    mode="elements",  # 保留结构
                    strategy="hi_res"  # 高精度
                )
                docs = loader.load()

            elif file_path.suffix.lower() == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
                docs = loader.load()

            elif file_path.suffix.lower() == '.xmind':
                docs = load_xmind(str(file_path))

            # 添加文件名到 metadata（便于溯源）
            for doc in docs:
                doc.metadata["source"] = str(file_path.name)

            all_docs.extend(docs)

        except Exception as e:
            print(f"❌ 加载失败 {file_path}: {e}")

    print(f"✅ 共加载 {len(all_docs)} 个文档片段")
    return all_docs