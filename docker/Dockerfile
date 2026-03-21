# Dockerfile
# Cloud Run 部署镜像
# ============================================================

FROM python:3.11-slim

# 工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements_serving.txt .
RUN pip install --no-cache-dir -r requirements_serving.txt

# 复制项目代码
COPY models/    ./models/
COPY serving/   ./serving/
COPY configs/   ./configs/

# 复制模型文件（从 GCS 挂载或直接 COPY）
# 注意: 模型文件较大，建议从 GCS 在启动时下载
# 如果直接打包进镜像（适合小模型）:
# COPY model_artifacts/ ./model_artifacts/

# Cloud Run 使用环境变量 PORT
ENV PORT=8080
ENV BASE_DIR=/app/model_artifacts

# 启动命令
CMD uvicorn serving.serving_api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1
