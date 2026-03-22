# models/recall/recall_evaluator.py
# 召回效果评估 — KuaiRec 版本
# 对应 Step2 Cell 10 + Cell 11
# ============================================================

import logging
import numpy as np
import pandas as pd
import faiss
from typing import Dict, List

logger = logging.getLogger(__name__)


class RecallEvaluator:
    """
    召回效果评估器

    KuaiRec 评估说明:
        small_matrix 密度 99.6%，每用户正样本数 ~3314
        Recall@K 天然偏低（分母大），Hit@K 更有参考意义
        NDCG@K 反映排序质量，是主要参考指标

    指标:
        Recall@K: 被召回的正样本占该用户全部正样本的比例
        Hit@K:    至少召回 1 个正样本的用户比例
        NDCG@K:   归一化折损累积增益，衡量排序质量
    """

    def evaluate(
        self,
        test_df:       pd.DataFrame,
        user_embs:     np.ndarray,       # [N_users, D]
        item_embs:     np.ndarray,       # [N_items, D]
        user_ids:      List,
        item_ids:      List,
        faiss_index,                      # FaissIndex 实例
        ks:            List[int] = [10, 20, 50],
        n_eval:        int = 500,
    ) -> Dict[str, float]:
        """
        Args:
            test_df:     测试集（含 user_id, item_id, label）
            user_embs:   用户 Embedding 矩阵
            item_embs:   视频 Embedding 矩阵
            user_ids:    与 user_embs 行对应的用户 ID
            item_ids:    与 item_embs 行对应的视频 ID
            faiss_index: FaissIndex 实例
            ks:          评估的 K 值列表
            n_eval:      评估用户数（为了速度，默认取前500个）

        Returns:
            指标字典 {'Recall@10': ..., 'Hit@20': ..., ...}
        """
        user_pos  = (test_df[test_df['label'] == 1]
                     .groupby('user_id')['item_id']
                     .apply(set).to_dict())
        uid2row   = {uid: i for i, uid in enumerate(user_ids)}
        eval_users = [u for u in list(user_pos.keys())[:n_eval]
                      if u in uid2row]

        results = {f'Recall@{k}': [] for k in ks}
        results.update({f'Hit@{k}':    [] for k in ks})
        results.update({f'NDCG@{k}':   [] for k in ks})

        for uid in eval_users:
            pos_items = user_pos.get(uid, set())
            if not pos_items:
                continue
            u_vec     = user_embs[uid2row[uid]:uid2row[uid]+1].astype(np.float32)
            retrieved = faiss_index.search_one(u_vec[0], top_k=max(ks))

            for k in ks:
                top_k = set(retrieved[:k])
                hits  = top_k & pos_items

                results[f'Recall@{k}'].append(
                    len(hits) / len(pos_items))
                results[f'Hit@{k}'].append(
                    1.0 if hits else 0.0)

                # NDCG
                dcg  = sum(1/np.log2(i+2)
                           for i, v in enumerate(retrieved[:k])
                           if v in pos_items)
                idcg = sum(1/np.log2(i+2)
                           for i in range(min(len(pos_items), k)))
                results[f'NDCG@{k}'].append(
                    dcg/idcg if idcg > 0 else 0)

        return {k: float(np.mean(v)) for k, v in results.items()}

    def compare_baselines(
        self,
        test_df:   pd.DataFrame,
        item_ids:  List,
        ks:        List[int] = [10, 20, 50],
        n_eval:    int = 500,
    ) -> Dict[str, Dict]:
        """
        对比 Baseline（随机召回 vs 热门召回）

        Returns:
            {'random': {...}, 'popular': {...}}
        """
        user_pos  = (test_df[test_df['label'] == 1]
                     .groupby('user_id')['item_id']
                     .apply(set).to_dict())
        eval_users = list(user_pos.keys())[:n_eval]
        pop_items  = test_df['item_id'].value_counts().index.tolist()

        baselines = {'random': {}, 'popular': {}}
        for strategy in ['random', 'popular']:
            recall_dict = {f'Recall@{k}': [] for k in ks}
            for uid in eval_users:
                pos_items = user_pos.get(uid, set())
                if not pos_items:
                    continue
                if strategy == 'random':
                    retrieved = np.random.choice(
                        item_ids, max(ks), replace=False).tolist()
                else:
                    retrieved = pop_items
                for k in ks:
                    hits = set(retrieved[:k]) & pos_items
                    recall_dict[f'Recall@{k}'].append(
                        len(hits) / len(pos_items))
            baselines[strategy] = {
                k: float(np.mean(v)) for k, v in recall_dict.items()}

        return baselines
