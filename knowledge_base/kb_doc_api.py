import json
import os
import urllib
from typing import List, Dict

from fastapi import Body, UploadFile, File, Form
from langchain_core.documents import Document

from db.repository.knowledge_base_repository import list_kbs_from_db
from db.repository.knowledge_file_repository import get_file_detail
from knowledge_base.kb_service.base import KBServiceFactory, get_kb_file_details
from knowledge_base.model.kb_document_model import DocumentWithVSId
from knowledge_base.utils import get_file_path, KnowledgeFile, files2docs_in_thread
from settings import Settings
from utils import build_logger, BaseResponse, get_default_embedding, ListResponse, run_in_thread_pool

logger = build_logger()




def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(
            ..., description="知识库名称", examples=["samples"]
        ),
        top_k: int = Body(3, description="匹配向量数"),
        score_threshold: float = Body(
            2.0,
            description="知识库匹配相关度阈值，取值范围在0-1之间，"
                        "SCORE越小，相关度越高，"
                        "取到2相当于不筛选，建议设置在0.5左右",
            ge=0.0,
            le=2.0,
        ),
        file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[Dict]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    data = []
    if kb is not None:
        if query:
            docs = kb.search_docs(query, top_k, score_threshold)
            # data = [DocumentWithVSId(**x[0].dict(), score=x[1], id=x[0].metadata.get("id")) for x in docs]
            data = [DocumentWithVSId(**{"id": x.metadata.get("id"), **x.dict()}) for x in docs]
        elif file_name or metadata:
            data = kb.list_docs(file_name=file_name, metadata=metadata)
            for d in data:
                if "vector" in d.metadata:
                    del d.metadata["vector"]
    return [x.dict() for x in data]


def upload_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        knowledge_base_name: str = Form(
            ..., description="知识库名称", examples=["samples"]
        ),
        override: bool = Form(False, description="覆盖已有文件"),
        to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
        chunk_size: int = Form(750, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(150, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(False, description="是否开启中文标题加强"),
        docs: str = Form("", description="自定义的docs，需要转为json字符串"),
        not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
):
    """
    API接口：上传文件，并/或向量化
    """
    # if not validate_kb_name(knowledge_base_name):
    #     return {"status": 'Fail', "message": "Don not attack me", "data": None}

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return {"status": 'Fail', "message": "未找到知识库", "data": None}

    docs = json.loads(docs) if docs else {}
    failed_files = {}
    file_names = list(docs.keys())

    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(
            files, knowledge_base_name=knowledge_base_name, override=override
    ):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        # failed_files.update(result['data']['failed_files'])
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(
        code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files}
    )


def _save_files_in_thread(
        files: List[UploadFile], knowledge_base_name: str, override: bool
):
    """
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        """
        保存单个文件。
        """
        try:
            filename = file.filename
            file_path = get_file_path(
                knowledge_base_name=knowledge_base_name, doc_name=filename
            )
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容
            if (
                    os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            return dict(code=500, msg=msg, data=data)

    params = [
        {"file": file, "knowledge_base_name": knowledge_base_name, "override": override}
        for file in files
    ]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


def delete_docs(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
        delete_content: bool = Body(False),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    # if not validate_kb_name(knowledge_base_name):
    #     return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{file_name} 文件删除失败，错误信息：{e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(
        code=200, msg=f"文件删除完成", data={"failed_files": failed_files}
    )


def update_docs(
        knowledge_base_name: str = Body(
            ..., description="知识库名称", examples=["samples"]
        ),
        file_names: List[str] = Body(
            ..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]
        ),
        chunk_size: int = Body(750, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(150, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(False, description="是否开启中文标题加强"),
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        docs: str = Body("", description="自定义的docs，需要转为json字符串"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
):
    """
    更新知识库文档
    """
    # if not validate_kb_name(knowledge_base_name):
    #     return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    kb_files = []
    docs = json.loads(docs) if docs else {}

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
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

    # 从文件生成docs，并进行向量化。
    # 这里利用了KnowledgeFile的缓存功能，在多线程中加载Document，然后传给KnowledgeFile
    for status, result in files2docs_in_thread(
            kb_files,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
    ):
        if status:
            kb_name, file_name, new_docs = result
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=True)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(
                filename=file_name, knowledge_base_name=knowledge_base_name
            )
            kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"为 {file_name} 添加自定义docs时出错：{e}"
            logger.error(f"{e.__class__.__name__}: {msg}")
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg="成功", data={"failed_files": failed_files})