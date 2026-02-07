"""
用户配置 Repository
"""
from sqlalchemy.orm import Session
from db.models.user_model import UserModel
from typing import Optional, Dict, Any


def get_user_by_identifier(session: Session, identifier: str) -> Optional[UserModel]:
    """根据用户标识符获取用户信息"""
    return session.query(UserModel).filter(UserModel.identifier == identifier).first()


def get_user_settings(session: Session, identifier: str) -> Dict[str, Any]:
    """
    获取用户的模型配置设置
    
    Args:
        session: 数据库会话
        identifier: 用户标识符 (如 'admin', 'user1')
        
    Returns:
        用户设置字典,如果没有则返回空字典
    """
    user = get_user_by_identifier(session, identifier)
    if user and user.metadata_:
        return user.metadata_.get("model_settings", {})
    return {}


def save_user_settings(session: Session, identifier: str, settings: Dict[str, Any]) -> bool:
    """
    保存用户的模型配置设置
    
    Args:
        session: 数据库会话
        identifier: 用户标识符 (如 'admin', 'user1')
        settings: 要保存的设置字典
        
    Returns:
        是否保存成功
    """
    try:
        user = get_user_by_identifier(session, identifier)
        if user:
            # 确保 metadata_ 是字典类型
            if not isinstance(user.metadata_, dict):
                user.metadata_ = {}
            
            # 更新 model_settings
            user.metadata_["model_settings"] = settings
            
            # 标记字段已修改(针对 JSON 类型字段)
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(user, "metadata_")
            
            session.commit()
            return True
        return False
    except Exception as e:
        session.rollback()
        print(f"保存用户设置失败: {e}")
        return False
