# 用户认证功能使用说明

## 概述

本项目实现了完整的用户注册、登录和认证系统，使用JWT（JSON Web Token）来管理用户会话状态。

## API端点

### 1. 获取图形验证码
- **URL**: `POST /api/auth/captcha`
- **功能**: 获取图形验证码
- **响应**:
```json
{
  "code": 200,
  "msg": "成功",
  "data": {
    "captcha_key": "captcha_...",
    "captcha_image": "data:image/png;base64,..."
  }
}
```

### 2. 验证图形验证码
- **URL**: `POST /api/auth/verify_captcha`
- **功能**: 验证用户输入的图形验证码
- **请求体**:
```json
{
  "captcha_key": "captcha_...",
  "captcha_code": "ABCD"
}
```

### 3. 发送邮箱验证码
- **URL**: `POST /api/auth/send_email_verification`
- **功能**: 向指定邮箱发送验证码
- **请求体**:
```json
{
  "email": "user@example.com"
}
```

### 4. 用户注册
- **URL**: `POST /api/auth/register`
- **功能**: 用户注册
- **请求体**:
```json
{
  "email": "user@example.com",
  "password": "password123",
  "confirm_password": "password123",
  "email_verification_code": "123456"
}
```
- **响应**:
```json
{
  "code": 200,
  "msg": "注册成功",
  "data": {
    "user_id": 1,
    "email": "user@example.com",
    "username": "user",
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
  }
}
```

### 5. 用户登录
- **URL**: `POST /api/auth/login`
- **功能**: 用户登录
- **请求体**:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```
- **响应**:
```json
{
  "code": 200,
  "msg": "登录成功",
  "data": {
    "user_id": 1,
    "email": "user@example.com",
    "username": "user",
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
    "token_type": "bearer"
  }
}
```

### 6. 刷新访问令牌
- **URL**: `POST /api/auth/refresh_token`
- **功能**: 使用刷新令牌获取新的访问令牌
- **请求体**:
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
}
```

### 7. 获取用户信息
- **URL**: `GET /api/auth/user_info`
- **功能**: 获取当前认证用户的信息
- **头部**: `Authorization: Bearer {access_token}`

## 前端集成示例

### 用户注册流程
```javascript
// 1. 获取图形验证码
const captchaResponse = await fetch('/api/auth/captcha', {
  method: 'GET'
});
const captchaData = await captchaResponse.json();

// 2. 验证图形验证码
const verifyResponse = await fetch('/api/auth/verify_captcha', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    captcha_key: captchaData.data.captcha_key,
    captcha_code: userInputCaptcha
  })
});

// 3. 发送邮箱验证码
await fetch('/api/auth/send_email_verification', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    email: 'user@example.com'
  })
});

// 4. 注册用户
const registerResponse = await fetch('/api/auth/register', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    email: 'user@example.com',
    password: 'password123',
    confirm_password: 'password123',
    email_verification_code: '123456'
  })
});

const registerData = await registerResponse.json();
if (registerData.code === 200) {
  // 保存token到本地存储
  localStorage.setItem('access_token', registerData.data.access_token);
  localStorage.setItem('refresh_token', registerData.data.refresh_token);
}
```

### 使用Token访问受保护的API
```javascript
// 使用token访问受保护的API
const token = localStorage.getItem('access_token');
const response = await fetch('/api/auth/user_info', {
  method: 'GET',
  headers: {
    'Authorization': `Bearer ${token}`
  }
});
```

### 自动刷新Token
```javascript
// 拦截请求并自动刷新过期的token
async function makeAuthenticatedRequest(url, options = {}) {
  let token = localStorage.getItem('access_token');
  
  // 在请求前检查token是否即将过期（这里简化处理）
  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      'Authorization': `Bearer ${token}`
    }
  });
  
  // 如果返回401错误，尝试刷新token
  if (response.status === 401) {
    const refreshToken = localStorage.getItem('refresh_token');
    const refreshResponse = await fetch('/api/auth/refresh_token', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ refresh_token: refreshToken })
    });
    
    const refreshData = await refreshResponse.json();
    if (refreshData.code === 200) {
      // 更新本地存储的token
      localStorage.setItem('access_token', refreshData.data.access_token);
      
      // 重新发起原始请求
      return fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          'Authorization': `Bearer ${refreshData.data.access_token}`
        }
      });
    } else {
      // 刷新失败，重定向到登录页面
      window.location.href = '/login';
    }
  }
  
  return response;
}
```

## 安全配置

### 环境变量设置
在`.env`文件中设置以下变量：
```env
# JWT密钥 - 请在生产环境中使用安全的随机密钥
SECRET_KEY=your-super-secret-key-change-this-in-production
```

### Token过期时间配置
- 访问令牌（Access Token）：30分钟
- 刷新令牌（Refresh Token）：7天

## 后端使用示例

### 在其他API端点中使用认证
```python
from chat.auth_middleware import get_current_active_user
from chat.token_manager import TokenData
from fastapi import Depends

async def protected_endpoint(current_user: TokenData = Depends(get_current_active_user)):
    """受保护的API端点示例"""
    return {
        "message": f"Hello user {current_user.user_id}",
        "email": current_user.email
    }
```

## 注意事项

1. **安全性**：请确保在生产环境中使用强密钥，并通过环境变量设置SECRET_KEY
2. **Token存储**：前端应安全地存储JWT token，建议使用HttpOnly Cookie或安全的本地存储
3. **过期处理**：实现自动token刷新机制，提升用户体验
4. **错误处理**：妥善处理认证失败的情况