import os
import re
import json
import traceback
from pathlib import Path
from typing import List, Dict
import chainlit as cl
from langchain_core.documents import Document

from settings import Settings
from knowledge_base.kb_utils import get_file_path, KnowledgeFile, files2docs_in_thread
from knowledge_base.kb_service.base import KBServiceFactory
from db.repository.knowledge_file_repository import get_file_detail
from utils import build_logger, BaseResponse

logger = build_logger()


async def upload_document(kb_name: str):
    print(f"[DEBUG] upload_document called with kb_name={kb_name}")
    files = await cl.AskFileMessage(
        content="请选择要上传的文件",
        accept={
            "text/plain": [".txt"],
            "application/pdf": [".pdf"],
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": [".docx"],
            "text/markdown": [".md", ".markdown"],
        },
        max_files=10,
        max_size_mb=20,
    ).send()

    if not files:
        print("[DEBUG] No files selected, returning")
        return
    
    print(f"[DEBUG] Received {len(files)} files: {[f.name for f in files]}")
    file_names = []
    async with cl.Step(name="文档上传与向量化") as step:
        try:
            step.output = f"正在保存 {len(files)} 个文件..."
            # 1. 保存文件到知识库目录
            for file in files:
                file_path = get_file_path(kb_name, file.name)
                if not file_path:
                    continue

                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # 优雅地保存文件内容
                content = getattr(file, "content", None)
                if content is None:
                    content = Path(file.path).read_bytes()

                Path(file_path).write_bytes(content)
                file_names.append(file.name)

            step.output = f"文件已保存，正在进行向量化处理..."
            print(f"[DEBUG] Starting vectorization for files: {file_names}")
            
            # 2. 调用公用向量化逻辑
            res = await cl.make_async(update_kb_docs)(
                knowledge_base_name=kb_name,
                file_names=file_names,
                override_custom_docs=True,
                chunk_size=Settings.kb_settings.CHUNK_SIZE,
                chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
                zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
            )
            
            print(f"[DEBUG] Vectorization result: code={res.code}, msg={res.msg}")

            if res.code == 200:
                failed = res.data.get("failed_files", {})
                if not failed:
                    step.output = f"✅ 成功：{len(files)} 个文件已上传并向量化到知识库 '{kb_name}'。"
                else:
                    step.output = f"⚠️ 处理完成，但部分文件失败: {', '.join(failed.keys())}"
            else:
                step.output = f"❌ 上传失败: {res.msg}"

        except Exception as e:
            error_detail = traceback.format_exc()
            print(f"[DEBUG] Error in upload_document: {e}")
            print(f"[DEBUG] Traceback: {error_detail}")
            step.output = f"❌ 处理过程中发生错误: {str(e)}"
            await cl.Message(content=f"详细错误堆栈:\n```\n{error_detail}\n```").send()

def _clean_document_content(doc: Document) -> Document:
    """清洗文档内容，移除控制字符和特殊 Unicode"""
    if not doc.page_content:
        return None
    
    content = doc.page_content
    # 移除控制字符 (0-31, 127)
    content = re.sub(r'[\x00-\x1f\x7f]', '', content)
    # 移除零宽字符等特殊 Unicode
    content = re.sub(r'[\u200b-\u200f\uFEFF\u202a-\u202e]', '', content)
    content = content.strip()
    
    if not content:
        return None
    
    doc.page_content = content
    return doc


def update_kb_docs(
        knowledge_base_name: str,
        file_names: List[str],
        chunk_size: int = 750,
        chunk_overlap: int = 150,
        zh_title_enhance: bool = False,
        override_custom_docs: bool = False,
        docs: str = "",
        not_refresh_vs_cache: bool = False,
) -> BaseResponse:
    """
    更新知识库文档
    """
    print(f"[DEBUG] update_kb_docs called: kb={knowledge_base_name}, files={file_names}")
    
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        print(f"[DEBUG] Knowledge base not found: {knowledge_base_name}")
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    kb_files = []
    docs_dict = json.loads(docs) if docs else {}

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs_dict:
            try:
                kb_files.append(
                    KnowledgeFile(
                        filename=file_name, knowledge_base_name=knowledge_base_name
                    )
                )
            except Exception as e:
                msg = f"加载文档 {file_name} 时出错：{e}"
                logger.error(f"{e.__class__.__name__}: {msg}")
                failed_files[file_name] = msg

    print(f"[DEBUG] Created {len(kb_files)} KnowledgeFile objects")
    
    # 收集所有文档，批量处理
    all_docs_to_add = []
    processed_count = 0
    
    # 从文件生成docs，并进行向量化
    for status, result in files2docs_in_thread(
            kb_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
    ):
        processed_count += 1
        print(f"[DEBUG] Processing file {processed_count}/{len(kb_files)}: status={status}")
        
        if status:
            kb_name, file_name, new_docs = result
            print(f"[DEBUG] File {file_name} parsed, got {len(new_docs)} docs")

            # 过滤无效段落并清洗文本
            valid_docs = [_clean_document_content(d) for d in new_docs]
            valid_docs = [d for d in valid_docs if d is not None]

            if not valid_docs:
                logger.warning(f"文件 {file_name} 向量化后没有有效文本段落，跳过。")
                continue

            # 设置文档 source 元数据
            for doc in valid_docs:
                doc.metadata.setdefault("source", file_name)
            
            all_docs_to_add.extend(valid_docs)
            print(f"[DEBUG] Added {len(valid_docs)} valid docs, total={len(all_docs_to_add)}")
        else:
            kb_name, file_name, error = result
            print(f"[DEBUG] File {file_name} failed: {error}")
            failed_files[file_name] = error

    # 批量添加所有文档到向量库（一次性调用，减少 IO 次数）
    if all_docs_to_add:
        try:
            kb.do_add_doc(all_docs_to_add)
            # 批量记录文件到数据库
            for file_name in file_names:
                if file_name not in failed_files:
                    try:
                        kb_file = KnowledgeFile(
                            filename=file_name, knowledge_base_name=knowledge_base_name
                        )
                        from db.repository.knowledge_file_repository import add_file_to_db
                        add_file_to_db(kb_file, custom_docs=False, docs_count=0, doc_infos=[])
                    except Exception as e:
                        logger.warning(f"记录文件 {file_name} 到数据库时出错: {e}")
        except Exception as e:
            logger.error(f"批量添加文档时出错: {e}")
            # 如果批量添加失败，回退到逐个添加
            for doc in all_docs_to_add:
                try:
                    kb.do_add_doc([doc])
                except Exception as e2:
                    logger.error(f"单个文档添加失败: {e2}")

    # 将自定义的docs进行向量化
    custom_docs_to_add = []
    for file_name, v in docs_dict.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            # 清洗自定义 docs
            cleaned_v = [_clean_document_content(d) for d in v]
            cleaned_v = [d for d in cleaned_v if d is not None]
            
            if cleaned_v:
                for doc in cleaned_v:
                    doc.metadata.setdefault("source", file_name)
                custom_docs_to_add.extend(cleaned_v)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            failed_files[file_name] = msg
    
    # 批量添加自定义文档
    if custom_docs_to_add:
        try:
            kb.do_add_doc(custom_docs_to_add)
        except Exception as e:
            logger.error(f"批量添加自定义文档时出错: {e}")

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg="成功", data={"failed_files": failed_files})
