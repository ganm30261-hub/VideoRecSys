# models/recall/faiss_index.py
# FAISS 向量索引构建与检索 — KuaiRec 版本
# 对应 Step2 Cell 9
# ============================================================

import os
import logging
import numpy as np
import faiss
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class FaissIndex:
    """
    FAISS ANN 向量检索封装

    对应企业: Milvus / 阿里云 Proxima 向量检索服务

    索引策略:
        视频数 <= 10,000  → IndexFlatIP（精确搜索，KuaiRec small_matrix用这个）
        视频数 > 10,000   → IndexIVFFlat（近似搜索，速度快）
    """

    def __init__(self, emb_dim: int, n_items: int):
        self.emb_dim = emb_dim
        self.n_items = n_items
        self.index   = None
        self.item_ids: List = []

    def build(
        self,
        item_embs:  np.ndarray,   # [N, D] float32
        item_ids:   List,
        use_ivf:    bool = False,  # True 用近似搜索（大规模场景）
        n_clusters: int  = 100,
    ) -> None:
        """
        构建 FAISS 索引

        Args:
            item_embs:  视频 Embedding 矩阵（L2 归一化后）
            item_ids:   与 item_embs 行对应的视频 ID 列表
            use_ivf:    是否使用 IVF 近似搜索
            n_clusters: IVF 聚类数（use_ivf=True 时有效）
        """
        assert item_embs.shape[0] == len(item_ids), \
            "item_embs 行数与 item_ids 长度不一致"

        embs = item_embs.astype(np.float32).copy()
        faiss.normalize_L2(embs)

        if use_ivf and len(item_ids) > 10000:
            logger.info(f"构建 IVFFlat 索引  n_clusters={n_clusters}")
            quantizer  = faiss.IndexFlatIP(self.emb_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.emb_dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
            self.index.train(embs)
        else:
            logger.info("构建 IndexFlatIP 精确索引")
            self.index = faiss.IndexFlatIP(self.emb_dim)

        self.index.add(embs)
        self.item_ids = list(item_ids)
        logger.info(f"FAISS 索引构建完成: {self.index.ntotal:,} 个向量")

    def search(
        self,
        user_embs: np.ndarray,  # [B, D] float32
        top_k:     int = 200,
    ) -> Tuple[np.ndarray, List[List]]:
        """
        批量检索

        Args:
            user_embs: 用户 Embedding 矩阵（会自动 L2 归一化）
            top_k:     每个用户返回的候选数

        Returns:
            scores:    [B, top_k] 相似度分数
            item_ids:  [B, top_k] 对应的视频 ID 列表
        """
        assert self.index is not None, "请先调用 build() 构建索引"
        embs = user_embs.astype(np.float32).copy()
        faiss.normalize_L2(embs)

        scores, indices = self.index.search(embs, top_k)
        result_ids = [
            [self.item_ids[i] for i in row if i >= 0]
            for row in indices
        ]
        return scores, result_ids

    def search_one(
        self,
        user_emb: np.ndarray,  # [D] float32
        top_k:    int = 200,
    ) -> List:
        """单用户检索（在线推理用）"""
        _, item_ids = self.search(user_emb[np.newaxis, :], top_k)
        return item_ids[0]

    def save(self, save_dir: str) -> None:
        """保存索引和 item_ids"""
        os.makedirs(save_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(save_dir, 'faiss_item.index'))
        np.save(os.path.join(save_dir, 'item_ids.npy'), np.array(self.item_ids))
        logger.info(f"FAISS 索引保存 → {save_dir}")

    def load(self, save_dir: str) -> None:
        """从磁盘加载索引"""
        self.index    = faiss.read_index(os.path.join(save_dir, 'faiss_item.index'))
        self.item_ids = np.load(os.path.join(save_dir, 'item_ids.npy')).tolist()
        logger.info(
            f"FAISS 索引加载: {self.index.ntotal:,} 个向量 ← {save_dir}")
