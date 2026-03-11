# ============================================================
# Dockerfile — 推理服务容器
# 用于 Step 4 完成后部署到 Google Cloud Run
# ============================================================

FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖（先复制 requirements 利用缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir \
    pandas numpy pyarrow fastapi uvicorn pydantic \
    torch --index-url https://download.pytorch.org/whl/cpu \
    faiss-cpu redis pyyaml loguru

# 复制项目代码
COPY serving/     ./serving/
COPY models/      ./models/
COPY pipeline/    ./pipeline/
COPY configs/     ./configs/

# 暴露端口
EXPOSE 8080

# 启动命令
CMD ["uvicorn", "serving.serving_api:app", "--host", "0.0.0.0", "--port", "8080"]
