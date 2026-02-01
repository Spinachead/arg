from typing import Dict, List
from db.models.knowledge_base_model import KnowledgeBaseModel
from db.models.knowledge_file_model import FileDocModel, KnowledgeFileModel
from db.session import with_session
from knowledge_base.kb_utils import KnowledgeFile
import uuid


@with_session
def list_file_num_docs_id_by_kb_name_and_file_name(
    session,
    kb_name: str,
    file_name: str,
) -> List[int]:
    """
    列出某知识库某文件对应的所有Document的id。
    返回形式：[str, ...]
    """
    doc_ids = (
        session.query(FileDocModel.docId)
        .filter_by(kbName=kb_name, fileName=file_name)
        .all()
    )
    return [int(_id[0]) for _id in doc_ids]


@with_session
def list_docs_from_db(
    session,
    kb_name: str,
    file_name: str = None,
    metadata: Dict = {},
) -> List[Dict]:
    """
    列出某知识库某文件对应的所有Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    """
    docs = session.query(FileDocModel).filter(FileDocModel.kbName.ilike(kb_name))
    if file_name:
        docs = docs.filter(FileDocModel.fileName.ilike(file_name))
    for k, v in metadata.items():
        docs = docs.filter(FileDocModel.metadata_[k].as_string() == str(v))

    return [{"id": x.docId, "metadata": x.metadata_} for x in docs.all()]


@with_session
def delete_docs_from_db(
    session,
    kb_name: str,
    file_name: str = None,
) -> List[Dict]:
    """
    删除某知识库某文件对应的所有Document，并返回被删除的Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    """
    docs = list_docs_from_db(kb_name=kb_name, file_name=file_name)
    query = session.query(FileDocModel).filter(FileDocModel.kbName.ilike(kb_name))
    if file_name:
        query = query.filter(FileDocModel.fileName.ilike(file_name))
    query.delete(synchronize_session=False)
    session.commit()
    return docs


@with_session
def add_docs_to_db(session, kb_name: str, file_name: str, doc_infos: List[Dict]):
    """
    将某知识库某文件对应的所有Document信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict}, ...]
    """
    # ! 这里会出现doc_infos为None的情况，需要进一步排查
    if doc_infos is None:
        print(
            "输入的server.db.repository.knowledge_file_repository.add_docs_to_db的doc_infos参数为None"
        )
        return False
    for d in doc_infos:
        obj = FileDocModel(
            id=str(uuid.uuid4()),
            kbName=kb_name,
            fileName=file_name,
            docId=d["id"],
            metadata_=d["metadata"],
        )
        session.add(obj)
    return True


@with_session
def count_files_from_db(session, kb_name: str) -> int:
    return (
        session.query(KnowledgeFileModel)
        .filter(KnowledgeFileModel.kbName.ilike(kb_name))
        .count()
    )


@with_session
def list_files_from_db(session, kb_name):
    files = (
        session.query(KnowledgeFileModel)
        .filter(KnowledgeFileModel.kbName.ilike(kb_name))
        .all()
    )
    docs = [f.fileName for f in files]
    return docs


@with_session
def add_file_to_db(
    session,
    kb_file: KnowledgeFile,
    docs_count: int = 0,
    custom_docs: bool = False,
    doc_infos: List[Dict] = [],  # 形式：[{"id": str, "metadata": dict}, ...]
):
    kb = session.query(KnowledgeBaseModel).filter_by(kbName=kb_file.kb_name).first()
    if kb:
        # 如果已经存在该文件，则更新文件信息与版本号
        existing_file: KnowledgeFileModel = (
            session.query(KnowledgeFileModel)
            .filter(
                KnowledgeFileModel.kbName.ilike(kb_file.kb_name),
                KnowledgeFileModel.fileName.ilike(kb_file.filename),
            )
            .first()
        )
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()

        if existing_file:
            existing_file.fileMtime = mtime
            existing_file.fileSize = size
            existing_file.docsCount = docs_count
            existing_file.customDocs = custom_docs
            existing_file.fileVersion += 1
        # 否则，添加新文件
        else:
            new_file = KnowledgeFileModel(
                id=str(uuid.uuid4()),
                fileName=kb_file.filename,
                fileExt=kb_file.ext,
                kbName=kb_file.kb_name,
                documentLoaderName=kb_file.document_loader_name,
                textSplitterName=kb_file.text_splitter_name or "SpacyTextSplitter",
                fileMtime=mtime,
                fileSize=size,
                docsCount=docs_count,
                customDocs=custom_docs,
            )
            kb.fileCount += 1
            session.add(new_file)
        add_docs_to_db(
            kb_name=kb_file.kb_name, file_name=kb_file.filename, doc_infos=doc_infos
        )
    return True


@with_session
def delete_file_from_db(session, kb_file: KnowledgeFile):
    existing_file = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.fileName.ilike(kb_file.filename),
            KnowledgeFileModel.kbName.ilike(kb_file.kb_name),
        )
        .first()
    )
    if existing_file:
        session.delete(existing_file)
        delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        session.commit()

        kb = (
            session.query(KnowledgeBaseModel)
            .filter(KnowledgeBaseModel.kbName.ilike(kb_file.kb_name))
            .first()
        )
        if kb:
            kb.fileCount -= 1
            session.commit()
    return True


@with_session
def delete_files_from_db(session, knowledge_base_name: str):
    session.query(KnowledgeFileModel).filter(
        KnowledgeFileModel.kbName.ilike(knowledge_base_name)
    ).delete(synchronize_session=False)
    session.query(FileDocModel).filter(
        FileDocModel.kbName.ilike(knowledge_base_name)
    ).delete(synchronize_session=False)
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kbName.ilike(knowledge_base_name))
        .first()
    )
    if kb:
        kb.fileCount = 0

    session.commit()
    return True


@with_session
def file_exists_in_db(session, kb_file: KnowledgeFile):
    existing_file = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.fileName.ilike(kb_file.filename),
            KnowledgeFileModel.kbName.ilike(kb_file.kb_name),
        )
        .first()
    )
    return True if existing_file else False


@with_session
def get_file_detail(session, kb_name: str, filename: str) -> dict:
    file: KnowledgeFileModel = (
        session.query(KnowledgeFileModel)
        .filter(
            KnowledgeFileModel.fileName.ilike(filename),
            KnowledgeFileModel.kbName.ilike(kb_name),
        )
        .first()
    )
    if file:
        return {
            "kbName": file.kbName,
            "fileName": file.fileName,
            "fileExt": file.fileExt,
            "fileVersion": file.fileVersion,
            "documentLoader": file.documentLoaderName,
            "textSplitter": file.textSplitterName,
            "createdAt": file.createdAt,
            "fileMtime": file.fileMtime,
            "fileSize": file.fileSize,
            "customDocs": file.customDocs,
            "docsCount": file.docsCount,
        }
    else:
        return {}
