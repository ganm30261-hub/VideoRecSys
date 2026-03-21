#!/bin/bash
# serving/start_server.sh
# 本地启动推理服务

set -e

# 模型文件目录（改成你的实际路径）
export BASE_DIR="${BASE_DIR:-/content/VideoRecSys}"
export PORT="${PORT:-8000}"

echo "================================"
echo "  🎬 VideoRecSys 推理服务启动"
echo "================================"
echo "  BASE_DIR = $BASE_DIR"
echo "  PORT     = $PORT"
echo ""

# 检查模型文件是否存在
required_files=(
    "$BASE_DIR/models/recall/faiss_item.index"
    "$BASE_DIR/models/recall/id_mappings.pkl"
    "$BASE_DIR/models/ranking/din_best.pt"
    "$BASE_DIR/models/ranking/scalers.pkl"
    "$BASE_DIR/feature_store/user_embeddings.parquet"
    "$BASE_DIR/feature_store/item_embeddings.parquet"
)

echo "检查模型文件..."
for f in "${required_files[@]}"; do
    if [ -f "$f" ]; then
        echo "  ✅ $f"
    else
        echo "  ❌ 缺少文件: $f"
        exit 1
    fi
done

echo ""
echo "启动 FastAPI 服务..."
echo "  访问地址: http://localhost:$PORT"
echo "  API 文档: http://localhost:$PORT/docs"
echo "  健康检查: http://localhost:$PORT/health"
echo ""

uvicorn serving.serving_api:app \
    --host 0.0.0.0 \
    --port $PORT \
    --workers 1
