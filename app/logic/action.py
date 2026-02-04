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


async def upload_document():
    files = await cl.AskFileMessage(
        content="请选择要上传的文件",
        accept=[
            "text/plain",
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/markdown",
        ],
        max_files=10,
    ).send()

    if not files:
        return

    kb_name = Settings.kb_settings.DEFAULT_KNOWLEDGE_BASE
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
            # 2. 调用公用向量化逻辑
            res = await cl.make_async(update_kb_docs)(
                knowledge_base_name=kb_name,
                file_names=file_names,
                override_custom_docs=True,
                chunk_size=Settings.kb_settings.CHUNK_SIZE,
                chunk_overlap=Settings.kb_settings.OVERLAP_SIZE,
                zh_title_enhance=Settings.kb_settings.ZH_TITLE_ENHANCE,
            )

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
            print(error_detail)
            step.output = f"❌ 处理过程中发生错误: {str(e)}"
            # 发送一个独立的错误消息以便用户查看详情
            await cl.Message(content=f"详细错误堆栈:\n```\n{error_detail}\n```").send()

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
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
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

    # 从文件生成docs，并进行向量化
    for status, result in files2docs_in_thread(
            kb_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
    ):
        if status:
            kb_name, file_name, new_docs = result

            # 过滤无效段落并清洗文本，防止 Ollama Embedding 产生 NaN 报错
            valid_docs = []
            for d in new_docs:
                if not d.page_content:
                    continue
                # 更彻底地清洗文本：移除所有控制字符、零宽字符、以及非 UTF-8 兼容字符
                content = d.page_content
                # 移除控制字符 (0-31, 127)
                content = re.sub(r'[\x00-\x1f\x7f]', '', content)
                # 移除零宽字符等特殊 Unicode
                content = re.sub(r'[\u200b-\u200f\uFEFF\u202a-\u202e]', '', content)
                content = content.strip()
                
                if content:
                    d.page_content = content
                    valid_docs.append(d)

            if not valid_docs:
                logger.warning(f"文件 {file_name} 向量化后没有有效文本段落，跳过。")
                continue

            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            # 显式传递 docs=valid_docs，确保底层直接使用过滤后的内容
            kb.update_doc(kb_file, docs=valid_docs, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    for file_name, v in docs_dict.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            # 同样对自定义 docs 进行强力清洗和过滤
            cleaned_v = []
            for d in v:
                if d.page_content:
                    content = re.sub(r'[\x00-\x1f\x7f]', '', d.page_content)
                    content = re.sub(r'[\u200b-\u200f\uFEFF\u202a-\u202e]', '', content).strip()
                    if content:
                        d.page_content = content
                        cleaned_v.append(d)
            if not cleaned_v:
                continue
                
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            kb.update_doc(kb_file, docs=cleaned_v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg="成功", data={"failed_files": failed_files})
