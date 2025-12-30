from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .token_manager import TokenManager, TokenData


security = HTTPBearer()


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> TokenData:
    """
    依赖项：获取当前认证用户
    """
    token = credentials.credentials
    token_data = TokenManager.verify_token(token)
    
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return token_data


def get_current_active_user(current_user: TokenData = Depends(get_current_user)):
    """
    依赖项：获取当前活跃用户
    """
    # 这里可以添加额外的用户验证逻辑，如检查用户是否被禁用等
    return current_user