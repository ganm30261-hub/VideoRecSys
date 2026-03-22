# serving/gcs_loader.py
# 启动时从 GCS 下载模型文件
# ============================================================

import os
import logging
from google.cloud import storage

log = logging.getLogger('GCSLoader')


def download_model_artifacts(
    bucket_name: str,
    gcs_prefix:  str,
    local_dir:   str,
) -> None:
    """
    从 GCS 下载模型文件到本地

    Args:
        bucket_name: GCS bucket 名称
        gcs_prefix:  GCS 路径前缀，如 VideoRecSys/model_artifacts
        local_dir:   本地保存目录，如 /app/model_artifacts
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # 需要下载的文件列表
    files = [
        'models/recall/faiss_item.index',
        'models/recall/id_mappings.pkl',
        'models/recall/item_ids.npy',
        'models/recall/user_ids.npy',
        'models/ranking/din_best.pt',
        'models/ranking/scalers.pkl',
        'feature_store/user_embeddings.parquet',
        'feature_store/item_embeddings.parquet',
        'data/features/user_features.parquet',
        'data/features/item_features.parquet',
        'data/features/user_sequences.parquet',
    ]

    log.info(f"从 GCS 下载模型文件  bucket={bucket_name}  prefix={gcs_prefix}")

    for file_path in files:
        gcs_path   = f"{gcs_prefix}/{file_path}"
        local_path = os.path.join(local_dir, file_path)

        # 创建本地目录
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # 已存在则跳过
        if os.path.exists(local_path):
            log.info(f"  已存在，跳过: {file_path}")
            continue

        log.info(f"  下载: {gcs_path} → {local_path}")
        blob = bucket.blob(gcs_path)
        blob.download_to_filename(local_path)

    log.info("✅ 模型文件下载完成")
