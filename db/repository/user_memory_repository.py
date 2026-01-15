from typing import List, Optional, Dict
from datetime import datetime
from sqlalchemy import desc
from sqlalchemy.orm import Session

from db.models.user_memory_model import UserMemoryModel
from db.session import with_session


@with_session
def add_user_memory(
    session: Session,
    user_id: int,
    memory_text: str,
    importance: int = 1,
    meta_data: dict = None,
) -> int:
    """
    添加用户记忆
    :param session: 数据库会话
    :param user_id: 用户ID
    :param memory_text: 记忆内容摘要
    :param importance: 重要程度 1-5
    :param meta_data: 元数据（JSON）
    :return: 记忆ID
    """
    memory = UserMemoryModel(
        user_id=user_id,
        memory_text=memory_text,
        importance=importance,
        meta_data=meta_data or {},
    )
    session.add(memory)
    session.flush()
    return memory.id


@with_session
def list_user_memories(
    session: Session,
    user_id: int,
    limit: int = 20,
) -> List[UserMemoryModel]:
    """
    获取用户记忆列表，按重要程度和最近使用时间排序
    :param session: 数据库会话
    :param user_id: 用户ID
    :param limit: 返回数量限制
    :return: 记忆列表
    """
    memories = (
        session.query(UserMemoryModel)
        .filter(UserMemoryModel.user_id == user_id)
        .order_by(
            desc(UserMemoryModel.importance),
            desc(UserMemoryModel.last_used_time)
        )
        .limit(limit)
        .all()
    )
    return memories


@with_session
def get_user_memory_by_id(
    session: Session,
    memory_id: int,
) -> Optional[UserMemoryModel]:
    """
    根据ID获取单条记忆
    :param session: 数据库会话
    :param memory_id: 记忆ID
    :return: 记忆对象或None
    """
    return session.query(UserMemoryModel).filter(UserMemoryModel.id == memory_id).first()


@with_session
def update_user_memory(
    session: Session,
    memory_id: int,
    memory_text: str = None,
    importance: int = None,
    meta_data: dict = None,
) -> bool:
    """
    更新用户记忆
    :param session: 数据库会话
    :param memory_id: 记忆ID
    :param memory_text: 新的记忆内容
    :param importance: 新的重要程度
    :param meta_data: 新的元数据
    :return: 是否更新成功
    """
    memory = session.query(UserMemoryModel).filter(UserMemoryModel.id == memory_id).first()
    if memory:
        if memory_text is not None:
            memory.memory_text = memory_text
        if importance is not None:
            memory.importance = importance
        if meta_data is not None:
            memory.meta_data = meta_data
        return True
    return False


@with_session
def update_memory_last_used(
    session: Session,
    memory_id: int,
) -> bool:
    """
    更新记忆的最近使用时间
    :param session: 数据库会话
    :param memory_id: 记忆ID
    :return: 是否更新成功
    """
    memory = session.query(UserMemoryModel).filter(UserMemoryModel.id == memory_id).first()
    if memory:
        memory.last_used_time = datetime.now()
        return True
    return False


@with_session
def delete_user_memory(
    session: Session,
    memory_id: int,
) -> bool:
    """
    删除用户记忆
    :param session: 数据库会话
    :param memory_id: 记忆ID
    :return: 是否删除成功
    """
    memory = session.query(UserMemoryModel).filter(UserMemoryModel.id == memory_id).first()
    if memory:
        session.delete(memory)
        return True
    return False


@with_session
def get_user_profile_from_memories(
    session: Session,
    user_id: int,
) -> Dict:
    """
    从用户记忆中提取用户画像/偏好信息
    :param session: 数据库会话
    :param user_id: 用户ID
    :return: 用户画像字典
    """
    memories = list_user_memories(user_id=user_id, limit=10)
    
    # 初始化用户画像
    profile = {
        "preferred_kbs": [],
        "preferred_domains": [],
        "tone": "标准",
        "recent_topics": [],
    }
    
    # 从 meta_data 中提取偏好信息
    kb_counts = {}
    domain_counts = {}
    
    for memory in memories:
        meta = memory.meta_data or {}
        
        # 统计知识库偏好
        kb = meta.get("kb_name")
        if kb:
            kb_counts[kb] = kb_counts.get(kb, 0) + memory.importance
        
        # 统计领域偏好
        domains = meta.get("domains", [])
        for domain in domains:
            domain_counts[domain] = domain_counts.get(domain, 0) + memory.importance
        
        # 收集风格偏好
        if "tone" in meta:
            profile["tone"] = meta["tone"]
        
        # 收集最近话题
        if memory.memory_text:
            profile["recent_topics"].append(memory.memory_text)
    
    # 按频率排序
    if kb_counts:
        profile["preferred_kbs"] = sorted(kb_counts.keys(), key=lambda k: kb_counts[k], reverse=True)[:3]
    if domain_counts:
        profile["preferred_domains"] = sorted(domain_counts.keys(), key=lambda k: domain_counts[k], reverse=True)[:3]
    
    return profile


@with_session
def create_memory_from_conversation(
    session: Session,
    user_id: int,
    conversation_summary: str,
    kb_name: str = None,
    domains: List[str] = None,
    importance: int = 3,
) -> int:
    """
    从对话历史创建记忆（供后台任务或定期汇总使用）
    :param session: 数据库会话
    :param user_id: 用户ID
    :param conversation_summary: 对话摘要
    :param kb_name: 使用的知识库名称
    :param domains: 涉及的领域列表
    :param importance: 重要程度
    :return: 记忆ID
    """
    meta_data = {}
    if kb_name:
        meta_data["kb_name"] = kb_name
    if domains:
        meta_data["domains"] = domains
    
    return add_user_memory(
        user_id=user_id,
        memory_text=conversation_summary,
        importance=importance,
        meta_data=meta_data,
    )
