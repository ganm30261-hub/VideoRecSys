# pipeline/feature_store.py
# 特征存储模块 — KuaiRec 版本
# 对应 Step1 Cell 9
# ============================================================

import os
import logging
import pickle
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    双层特征存储 — KuaiRec 版本

    在线层: Python dict 模拟 Redis
    离线层: Parquet 文件模拟 Hive/OSS

    KuaiRec 特有:
        - 额外存储 watch_ratio 统计分布（用于监控）
        - 支持 Embedding 读写（Step2 双塔产出）
    """

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self._cache: Dict[str, Any] = {}
        logger.info(f"FeatureStore 初始化  store_dir={store_dir}")

    # ─────────────────────────────────────────
    # 写入接口
    # ─────────────────────────────────────────

    def write_users(self, user_feat: pd.DataFrame) -> None:
        """写入用户特征"""
        path = os.path.join(self.store_dir, 'user_features.parquet')
        user_feat.to_parquet(path, index=False, compression='snappy')
        for _, row in user_feat.iterrows():
            self._cache[f"user:{row['user_id']}"] = row.to_dict()
        logger.info(f"写入用户特征: {len(user_feat):,} 用户 → {path}")

    def write_items(self, item_feat: pd.DataFrame) -> None:
        """写入视频特征"""
        path = os.path.join(self.store_dir, 'item_features.parquet')
        item_feat.to_parquet(path, index=False, compression='snappy')
        for _, row in item_feat.iterrows():
            self._cache[f"item:{row['item_id']}"] = row.to_dict()
        logger.info(f"写入视频特征: {len(item_feat):,} 视频 → {path}")

    def write_sequences(self, seq_df: pd.DataFrame) -> None:
        """写入用户行为序列"""
        path = os.path.join(self.store_dir, 'user_sequences.parquet')
        seq_df.to_parquet(path, index=False, compression='snappy')
        for _, row in seq_df.iterrows():
            self._cache[f"seq:{row['user_id']}"] = {
                'all_seq'    : row.get('all_seq', ''),
                'pos_seq'    : row.get('pos_seq', ''),
                'seq_len'    : row.get('seq_len', 0),
                'pos_seq_len': row.get('pos_seq_len', 0),
            }
        logger.info(f"写入用户序列: {len(seq_df):,} 用户 → {path}")

    def write_embeddings(
        self,
        emb_df: pd.DataFrame,
        entity: str,  # 'user' or 'item'
    ) -> None:
        """写入 Step2 双塔 Embedding"""
        path = os.path.join(self.store_dir, f'{entity}_embeddings.parquet')
        emb_df.to_parquet(path, index=False, compression='snappy')
        id_col   = f'{entity}_id'
        emb_cols = [c for c in emb_df.columns if c.startswith(f'{entity}_emb_')]
        for _, row in emb_df.iterrows():
            self._cache[f"emb:{entity}:{row[id_col]}"] = \
                row[emb_cols].values.astype(np.float32)
        logger.info(f"写入 {entity} Embedding: {len(emb_df):,} 条 → {path}")

    # ─────────────────────────────────────────
    # 读取接口
    # ─────────────────────────────────────────

    def get_user(self, user_id: int) -> Optional[Dict]:
        return self._cache.get(f"user:{user_id}")

    def get_item(self, item_id: int) -> Optional[Dict]:
        return self._cache.get(f"item:{item_id}")

    def get_sequence(self, user_id: int) -> Optional[Dict]:
        return self._cache.get(f"seq:{user_id}")

    def get_user_embedding(self, user_id: int) -> Optional[np.ndarray]:
        return self._cache.get(f"emb:user:{user_id}")

    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        return self._cache.get(f"emb:item:{item_id}")

    def batch_get_users(self, user_ids: List[int]) -> Dict[int, Dict]:
        return {uid: self.get_user(uid) for uid in user_ids
                if self.get_user(uid) is not None}

    def batch_get_items(self, item_ids: List[int]) -> Dict[int, Dict]:
        return {iid: self.get_item(iid) for iid in item_ids
                if self.get_item(iid) is not None}

    # ─────────────────────────────────────────
    # 持久化
    # ─────────────────────────────────────────

    def save_snapshot(self) -> None:
        """保存内存缓存快照"""
        path = os.path.join(self.store_dir, 'snapshot.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self._cache, f)
        size_mb = os.path.getsize(path) / 1024 ** 2
        logger.info(
            f"缓存快照保存: {len(self._cache):,} keys  "
            f"{size_mb:.2f} MB → {path}"
        )

    def load_snapshot(self) -> None:
        """从磁盘加载缓存快照"""
        path = os.path.join(self.store_dir, 'snapshot.pkl')
        if not os.path.exists(path):
            logger.warning(f"快照不存在: {path}")
            return
        with open(path, 'rb') as f:
            self._cache = pickle.load(f)
        logger.info(f"缓存快照加载: {len(self._cache):,} keys ← {path}")

    def stats(self) -> Dict[str, int]:
        return {
            'total_keys': len(self._cache),
            'user_keys' : sum(1 for k in self._cache if k.startswith('user:')),
            'item_keys' : sum(1 for k in self._cache if k.startswith('item:')),
            'seq_keys'  : sum(1 for k in self._cache if k.startswith('seq:')),
            'emb_keys'  : sum(1 for k in self._cache if k.startswith('emb:')),
        }
