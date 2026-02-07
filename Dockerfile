FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/data/knowledge_base /app/data/nltk_data /app/log

# 下载NLTK数据
RUN python -c "import nltk; nltk.download('punkt', download_dir='/app/data/nltk_data'); nltk.download('stopwords', download_dir='/app/data/nltk_data')"

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV CHATCHAT_ROOT=/app
ENV PYTHONUNBUFFERED=1

# 启动命令
CMD ["python", "-m", "chainlit", "run", "app/app.py", "-h", "0.0.0.0", "-w"]