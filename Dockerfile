FROM python:3.10-slim

WORKDIR /app

# 使用清华大学的 Debian 镜像源
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources || \
    sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list

# 安装系统依赖（包括 unstructured 所需的依赖）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制应用代码
COPY . .

# 创建必要的目录
RUN mkdir -p /app/app/data/knowledge_base /app/app/log /app/app/data/nltk_data

# 预下载 nltk 数据（使用国内镜像）
RUN python3 -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', download_dir='/app/app/data/nltk_data', quiet=True); nltk.download('punkt', download_dir='/app/app/data/nltk_data', quiet=True)" || echo "NLTK download failed, will retry at runtime"

# 预下载 Spacy 中文模型（避免运行时下载）
RUN python3 -m spacy download zh_core_web_sm || echo "Spacy model download failed, will use fallback"

# 预下载 RapidOCR 模型（避免运行时下载）
RUN python3 -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()" || echo "RapidOCR init failed, will retry at runtime"

# 暴露端口
EXPOSE 8000

# 设置环境变量
ENV CHATCHAT_ROOT=/app/app
ENV PYTHONUNBUFFERED=1
ENV NLTK_DATA=/app/app/data/nltk_data

# 启动命令
WORKDIR /app/app
CMD ["python", "-m", "chainlit", "run", "app.py", "--host", "0.0.0.0", "-w"]