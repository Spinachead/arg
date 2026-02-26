# ARG - 智能知识库助手

ARG 是一款基于大语言模型的企业级智能知识库问答系统，支持 RAG（检索增强生成）、SQL 数据库查询、多轮对话等功能。

## 功能特性

- **智能问答**：基于 RAG 技术，结合企业知识库提供精准回答
- **SQL 查询**：支持自然语言转 SQL，自动查询数据库
- **多轮对话**：支持上下文感知的连续对话
- **知识库管理**：支持多种文档格式（PDF、Word、Excel、图片等）的导入和管理
- **MCP 集成**：支持 Model Context Protocol，可扩展外部工具能力
- **用户认证**：支持密码登录和会话管理
- **流式输出**：实时响应，提升用户体验

## 技术架构

### 核心框架

| 组件 | 技术栈 | 说明 |
|------|--------|------|
| 前端界面 | [Chainlit](https://chainlit.io/) | Python 原生 AI 对话界面框架 |
| 工作流编排 | [LangGraph](https://langchain-ai.github.io/langgraph/) | 构建复杂 LLM 工作流 |
| LLM 调用 | [LangChain](https://langchain.com/) | 大语言模型统一接口 |
| 向量数据库 | [Qdrant](https://qdrant.tech/) / ChromaDB / FAISS | 文档向量存储与检索 |
| 关系数据库 | PostgreSQL | 用户数据、会话记录存储 |
| 文档解析 | unstructured + OCR | 多格式文档解析 |

### 主要依赖

```
Python 3.10+
chainlit >= 1.0.0
langchain >= 1.0.0
langgraph
qdrant-client
chromadb
sqlalchemy
fastapi
```

完整依赖列表见 [requirements.txt](./requirements.txt)

## 快速开始

### 环境要求

- Python 3.10+
- Docker & Docker Compose（推荐）
- PostgreSQL 15+
- Qdrant（向量数据库）

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd arg
```

### 2. 配置环境变量

复制环境变量模板：

```bash
cp app/.env.example app/.env
```

编辑 `app/.env` 文件，配置以下关键参数：

```env
# 数据库配置
DATABASE_URL=postgresql+asyncpg://postgres:your_password@localhost:5432/arg

# LLM API 密钥（至少配置一个）
DASHSCOPE_API_KEY=sk-your-dashscope-key      # 阿里云通义千问
DEEPSEEK_API_KEY=sk-your-deepseek-key          # DeepSeek
OPENAI_API_KEY=sk-your-openai-key              # OpenAI（可选）

# 默认模型
DEFAULT_LLM_MODEL=qwen-max                     # 或 deepseek-chat

# 认证密钥
CHAINLIT_AUTH_SECRET=your-secret-key
AUTH_SECRET_KEY=your-auth-secret

# 管理员密码
ADMIN_PASS=your-admin-password
```

### 3. 本地开发运行

#### 方式一：Conda 环境（推荐开发使用）

```bash
# 创建环境
conda create -n arg python=3.10
conda activate arg

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 设置项目根目录（Windows）
$env:CHATCHAT_ROOT="e:\pythonPro\arg\app"

# 设置项目根目录（Linux/Mac）
export CHATCHAT_ROOT=/path/to/arg/app

# 启动应用
cd app
python -m chainlit run app.py -w
```

应用将在 http://localhost:8000 启动，`-w` 参数支持热重载。

#### 方式二：Docker Compose（推荐生产使用）

```bash
# 构建并启动所有服务
docker-compose up -d --build

# 查看日志
docker-compose logs -f app

# 查看最近100行日志
docker-compose logs --tail=100 app
```

服务启动后访问 http://localhost:8000

## 生产部署

### 服务器环境要求

- Linux 服务器（Ubuntu 20.04+ / CentOS 7+）
- Docker 20.10+ 和 Docker Compose
- 域名（可选，用于 HTTPS）
- 宝塔面板（推荐，用于 Nginx 反向代理管理）

### 部署步骤

#### 1. 服务器准备

```bash
# 创建部署目录
mkdir -p /opt/arg_app
cd /opt/arg_app

# 上传项目文件（通过 git 或 scp）
git clone <your-repo-url> .
```

#### 2. 配置生产环境变量

```bash
# 编辑生产环境配置
vim app/.env
```

关键配置项：
```env
# 数据库使用 Docker 内网地址
DATABASE_URL=postgresql+asyncpg://postgres:postgres@db:5432/langchain_chatchat

# Qdrant 向量数据库地址
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# 生产环境关闭调试
LANGSMITH_TRACING=false
```

#### 3. 启动服务

```bash
docker-compose up -d
```


4. 申请 SSL 证书并开启 HTTPS

#### 4. 验证部署

```bash
# 检查容器状态
docker-compose ps

# 查看应用日志
docker logs -f arg_app
```

访问 `https://rag.yourdomain.com` 验证服务是否正常。

### Docker 镜像构建与推送（可选）

如需使用私有镜像仓库：

```bash
# 1. 登录镜像仓库
docker login your-registry.com

# 2. 构建镜像
docker build -t arg-app:latest .

# 3. 标记镜像
docker tag arg-app:latest your-registry.com/library/arg-app:latest

# 4. 推送镜像
docker push your-registry.com/library/arg-app:latest

# 5. 服务器拉取运行
docker pull your-registry.com/library/arg-app:latest
```

## 项目结构

```
arg/
├── app/                          # 应用主目录
│   ├── app.py                    # 程序入口
│   ├── core/                     # 核心逻辑
│   │   ├── state_graph/          # LangGraph 工作流
│   │   │   ├── nodes/            # 工作流节点
│   │   │   ├── states/           # 状态定义
│   │   │   ├── main_graph.py     # 主工作流
│   │   │   ├── knowledge_query_graph.py  # 知识库查询子图
│   │   │   └── sql_query_graph.py        # SQL 查询子图
│   │   └── prompts.py            # 提示词模板
│   ├── db/                       # 数据库相关
│   │   ├── models/               # SQLAlchemy 模型
│   │   └── repository/           # 数据访问层
│   ├── file_rag/                 # 文档处理与 RAG
│   │   ├── document_loaders/     # 文档加载器
│   │   ├── retrievers/           # 向量检索
│   │   └── text_splitter/        # 文本分割
│   ├── knowledge_base/           # 知识库管理
│   ├── logic/                    # 业务逻辑
│   ├── data/                     # 数据目录
│   └── .env                      # 环境变量
├── docker-compose.yml            # Docker 编排配置
├── Dockerfile                    # 应用镜像构建
└── requirements.txt              # Python 依赖
```

## 常用命令

```bash
# Docker 操作
docker-compose up -d              # 后台启动
docker-compose down               # 停止服务
docker-compose restart app        # 重启应用
docker-compose logs -f app        # 实时查看日志
docker exec -it arg_app bash      # 进入容器

# 数据库操作
docker exec -it arg_postgres psql -U postgres -d arg

# 本地开发
python -m chainlit run app.py -w  # 热重载模式
```

## 贡献指南

欢迎提交 Issue 和 Pull Request！

## 许可证

[MIT License](./LICENSE)
