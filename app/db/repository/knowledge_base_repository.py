from db.models.knowledge_base_model import KnowledgeBaseModel, KnowledgeBaseSchema
from db.session import with_session
import uuid


@with_session
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    # 创建知识库实例
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kbName.ilike(kb_name))
        .first()
    )
    if not kb:
        kb = KnowledgeBaseModel(
            id=str(uuid.uuid4()),
            kbName=kb_name, kbInfo=kb_info, vsType=vs_type, embedModel=embed_model
        )
        session.add(kb)
    else:  # update kb with new vsType and embedModel
        kb.kbInfo = kb_info
        kb.vsType = vs_type
        kb.embedModel = embed_model
    return True


@with_session
def list_kbs_from_db(session, min_file_count: int = -1):
    kbs = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.fileCount > min_file_count)
        .all()
    )
    kbs = [KnowledgeBaseSchema.model_validate(kb) for kb in kbs]
    return kbs


@with_session
def kb_exists(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kbName.ilike(kb_name))
        .first()
    )
    status = True if kb else False
    return status


@with_session
def load_kb_from_db(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kbName.ilike(kb_name))
        .first()
    )
    if kb:
        kb_name, vs_type, embed_model = kb.kbName, kb.vsType, kb.embedModel
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


@with_session
def delete_kb_from_db(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kbName.ilike(kb_name))
        .first()
    )
    if kb:
        session.delete(kb)
    return True


@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    kb: KnowledgeBaseModel = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kbName.ilike(kb_name))
        .first()
    )
    if kb:
        return {
            "kbName": kb.kbName,
            "kbInfo": kb.kbInfo,
            "vsType": kb.vsType,
            "embedModel": kb.embedModel,
            "fileCount": kb.fileCount,
            "createdAt": kb.createdAt,
        }
    else:
        return {}
