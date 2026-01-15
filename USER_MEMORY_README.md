# PostgreSQL 长期记忆实现说明

## 概述

基于 PostgreSQL 实现了用户长期记忆功能，可以根据用户的历史对话记录和偏好来个性化对话体验。

## 数据库结构

### user_memory 表

| 字段 | 类型 | 说明 |
|------|------|------|
| id | Integer | 主键ID |
| user_id | Integer | 用户ID（外键关联user表） |
| memory_text | String(1024) | 记忆内容摘要 |
| importance | Integer | 重要程度 1-5 |
| last_used_time | DateTime | 最近使用时间 |
| create_time | DateTime | 创建时间 |
| meta_data | JSON | 元数据（存储知识库偏好、领域、风格等） |

### meta_data 结构示例

```json
{
  "kb_name": "law_kb",
  "domains": ["law", "compliance"],
  "tone": "专业详细"
}
```

## 核心功能

### 1. Repository 函数

#### 基础操作
- `add_user_memory()` - 添加用户记忆
- `list_user_memories()` - 获取用户记忆列表（按重要程度和使用时间排序）
- `get_user_memory_by_id()` - 根据ID获取单条记忆
- `update_user_memory()` - 更新记忆内容
- `delete_user_memory()` - 删除记忆
- `update_memory_last_used()` - 更新最近使用时间

#### 高级功能
- `get_user_profile_from_memories()` - 从记忆中提取用户画像
- `create_memory_from_conversation()` - 从对话历史创建记忆

### 2. agent_chat 集成

#### 工作流程

```
用户请求 → 获取 user_id
    ↓
加载用户长期记忆（user_profile）
    ↓
注入到 generate_queries（影响知识库选择和查询改写）
    ↓
执行检索
    ↓
注入到最终 Prompt（影响回答风格）
    ↓
生成回答 + 保存 user_profile 到 meta_data
```

#### user_profile 结构

```python
{
    "preferred_kbs": ["law_kb", "company_docs"],     # 偏好知识库
    "preferred_domains": ["law", "product"],          # 偏好领域
    "tone": "简明扼要",                               # 回答风格
    "recent_topics": ["法律问题", "API使用"]          # 最近话题
}
```

## 使用方法

### 1. 创建数据库表

```bash
# 运行数据库初始化（会自动创建所有表包括 user_memory）
python init_database.py --create-tables
```

### 2. 添加用户记忆

```python
from db.repository.user_memory_repository import add_user_memory

# 方式1: 直接添加
memory_id = add_user_memory(
    user_id=1,
    memory_text="用户经常询问法律合规相关问题",
    importance=5,
    meta_data={
        "kb_name": "law_kb",
        "domains": ["law", "compliance"],
        "tone": "专业详细"
    }
)

# 方式2: 从对话历史创建
from db.repository.user_memory_repository import create_memory_from_conversation

memory_id = create_memory_from_conversation(
    user_id=1,
    conversation_summary="用户询问公司产品API使用方法",
    kb_name="company_docs",
    domains=["product", "api"],
    importance=4
)
```

### 3. 获取用户画像

```python
from db.repository.user_memory_repository import get_user_profile_from_memories

profile = get_user_profile_from_memories(user_id=1)
# 返回: {"preferred_kbs": [...], "preferred_domains": [...], "tone": "...", "recent_topics": [...]}
```

### 4. 调用 agent_chat

现在 agent_chat 需要用户认证，会自动加载并使用用户记忆：

```bash
# 调用时需要在 Header 中提供 JWT Token
curl -X POST "http://localhost:7861/api/agent_chat" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "请问劳动法相关的规定",
    "kb_name": "law_kb",
    "top_k": 3,
    "score_threshold": 0.5
  }'
```

## LangSmith Prompt 模板配置

为了让 user_profile 生效，需要在 LangSmith 的 Prompt 模板中添加变量：

```
你是一个专业的知识库助手。

用户偏好信息：
{user_profile}

请根据用户的偏好风格回答问题。

上下文信息：
{context}

参考来源：
{sources}

用户问题：
{question}
```

## 测试

运行测试脚本：

```bash
python test_user_memory.py
```

## 后续优化方向

### 1. 自动记忆生成
可以创建后台任务，定期分析用户的对话历史，自动生成或更新记忆：

```python
# 定期任务（伪代码）
def auto_generate_memories():
    # 1. 获取用户最近 N 条对话
    # 2. 调用 LLM 总结用户偏好
    # 3. 更新或创建 user_memory
    pass
```

### 2. 记忆衰减机制
根据 `last_used_time` 和 `importance` 实现记忆衰减：

```python
# 降低长时间未使用的低重要性记忆的权重
# 或定期清理过期记忆
```

### 3. 向量化记忆
将 `memory_text` 向量化，支持语义检索：

```python
# 为每条记忆生成 embedding
# 当用户提问时，先检索相关记忆再注入 Prompt
```

### 4. 多模态记忆
扩展支持图片、文件偏好等：

```python
meta_data = {
    "preferred_file_types": ["pdf", "docx"],
    "preferred_image_sources": ["official_diagrams"]
}
```

## 注意事项

1. **用户认证**：agent_chat 现在需要通过 JWT Token 认证，确保已登录
2. **性能考虑**：记忆查询会在每次对话时执行，如果用户记忆很多，考虑添加缓存
3. **隐私保护**：user_memory 包含用户敏感信息，注意数据安全
4. **Prompt 长度**：如果 user_profile 内容过多，注意 Prompt token 限制

## 相关文件

- `db/models/user_memory_model.py` - 数据模型
- `db/repository/user_memory_repository.py` - Repository 层
- `chat/kb_chat.py` - agent_chat 集成
- `api_server/chat_routes.py` - 路由配置
- `knowledge_base/migrate.py` - 数据库迁移
- `test_user_memory.py` - 测试脚本
