# arg
公司知识库助手

**常用命令：**
```bash
docker logs -f app
docker-compose logs --tail=100 app

```

**运行命令**
```bash
docker-compose build --no-cache
docker-compose up -d --build app
```

**设置根目录**
windows系统下执行
```bash
$env:CHATCHAT_ROOT="e:\pythonPro\arg\app"
```
linux系统下执行





**conda相关命令**
```bash
conda env list 查看所有环境
conda create -n myenv python=3.10 # 创建名为 myenv 的环境，指定 Python 版本
conda create -n myenv python=3.10 numpy pandas jupyter 同时安装包（推荐）
conda activate myenv 激活
conda deactivate 退出
conda env remove -n myenv 删除env环境
conda remove -n myenv --all 删除环境

cond install numpy 从默认channel安装
conda install -c conda-forge langchain-chroma 从指定 channel 安装（推荐使用 conda-forge）
conda update numpy
conda update --all
conda list
conda list | grep numpy  搜索特定包
conda list | findstr numpy 搜索特定包
conda remove numpy 卸载包
```


**热部署chainlit的方法**
```bash
conda activate torch_env
cd /d E:\pythonPro\arg\app
python -m chainlit run app.py -w

conda activate chatchat
cd /d D:\python\arg\app
python -m chainlit run app.py -w
```

**我是用的mcp**
```bash
npx -y @modelcontextprotocol/server-filesystem
npx -y @modelcontextprotocol/server-filesystem E:/pythonPro/mcp_server E:/pythonPro/arg
```

**docker打包镜像到Harbor**
```bash
# 1. 登录 Harbor (只需执行一次)
docker login 47.119.147.245:1683 要先进入docker deskstop设置 insecure-registries

# 2. 构建镜像 (基于你项目根目录的 Dockerfile)
docker build -t arg-app:latest .

# 3. 为镜像打上 Harbor 的 Tag
# 格式: docker tag [本地镜像名]:[标签] [Harbor地址]/[项目名]/[镜像名]:[标签]
docker tag arg-app:latest 47.119.147.245:1683/library/arg-app:v1.1

# 4. 推送到 Harbor
docker push 47.119.147.245:1683/library/arg-app:v1.1
# 5. 在服务器上运行
docker pull 127.0.0.1:1683/library/arg-app:v1.1
```

**本地代码更新后**
```bash
 docker build -t arg-app:v1.1 .
 docker tag arg-app:v1.1 47.119.147.245:1683/library/arg-app:v1.1
 docker push 47.119.147.245:1683/library/arg-app:v1.1
 ```
