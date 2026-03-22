# models/ranking/ranking_predictor.py
# 精排在线推理 — KuaiRec 版本
# 对应 Step3 Cell 8（完整推荐链路）
# ============================================================

import logging
import numpy as np
import torch
import faiss
from typing import List, Tuple, Dict

logger = logging.getLogger(__name__)


class RankingPredictor:
    """
    完整推荐链路在线推理器

    流程:
        输入用户 ID
        → 双塔 FAISS 召回 K 个候选
        → DIN 多任务精排（CTR + 完播率打分）
        → 融合排序（α*CTR + (1-α)*完播率）
        → 返回 Top N

    对应企业: 推理服务（Serving）层
    """

    def __init__(
        self,
        model,
        faiss_index,
        user_emb_matrix:  np.ndarray,
        user_id2row:      Dict,
        recall_item_ids:  List,
        user_feat_dict:   Dict,
        item_feat_dict:   Dict,
        user_emb_dict:    Dict,
        item_emb_dict:    Dict,
        seq_dict:         Dict,
        cfg:              dict,
    ):
        self.model           = model
        self.faiss_index     = faiss_index
        self.user_emb_matrix = user_emb_matrix.astype(np.float32)
        self.user_id2row     = user_id2row
        self.recall_item_ids = recall_item_ids
        self.u_feat          = user_feat_dict
        self.i_feat          = item_feat_dict
        self.u_emb           = user_emb_dict
        self.i_emb           = item_emb_dict
        self.seq_dict        = seq_dict
        self.cfg             = cfg
        self.device          = cfg['device']

        self.u_dim   = len(next(iter(user_feat_dict.values())))
        self.i_dim   = len(next(iter(item_feat_dict.values())))
        self.emb_dim = len(next(iter(user_emb_dict.values())))

        # L2 归一化用户 Embedding（FAISS 内积 = 余弦相似度）
        faiss.normalize_L2(self.user_emb_matrix)

    def recommend(
        self,
        user_id:  int,
        recall_k: int   = 200,
        top_n:    int   = 10,
        alpha:    float = 0.5,
    ) -> List[Tuple]:
        """
        单用户完整推荐链路

        Args:
            user_id:  目标用户 ID
            recall_k: 双塔召回候选数
            top_n:    最终返回推荐数
            alpha:    CTR 权重（1-alpha 为完播率权重）

        Returns:
            [(item_id, ctr_score, duration_score, final_score), ...]
        """
        self.model.eval()

        # ── 阶段1: 双塔召回 ──
        if user_id not in self.user_id2row:
            logger.warning(f"用户 {user_id} 不在 Embedding 中")
            return []

        u_vec    = self.user_emb_matrix[
            self.user_id2row[user_id]:self.user_id2row[user_id]+1]
        _, idxs  = self.faiss_index.search(u_vec, recall_k)
        cand_ids = [int(self.recall_item_ids[i]) for i in idxs[0]]

        # ── 阶段2: DIN 精排 ──
        u_feat_t = torch.tensor(
            self.u_feat.get(user_id, np.zeros(self.u_dim,  np.float32)),
            dtype=torch.float).unsqueeze(0).to(self.device)
        u_emb_t  = torch.tensor(
            self.u_emb.get(user_id, np.zeros(self.emb_dim, np.float32)),
            dtype=torch.float).unsqueeze(0).to(self.device)

        seq      = self.seq_dict.get(user_id, [])[-self.cfg['seq_len']:]
        pad_len  = self.cfg['seq_len'] - len(seq)
        seq_t    = torch.tensor(
            [0]*pad_len + seq, dtype=torch.long).unsqueeze(0).to(self.device)
        mask_t   = torch.tensor(
            [0]*pad_len + [1]*len(seq), dtype=torch.float).unsqueeze(0).to(self.device)

        scores = []
        batch  = 64
        for start in range(0, len(cand_ids), batch):
            b_iids = cand_ids[start:start+batch]
            B      = len(b_iids)

            i_feat_b = torch.tensor(
                np.stack([self.i_feat.get(i, np.zeros(self.i_dim,  np.float32))
                          for i in b_iids]),
                dtype=torch.float).to(self.device)
            i_emb_b  = torch.tensor(
                np.stack([self.i_emb.get(i, np.zeros(self.emb_dim, np.float32))
                          for i in b_iids]),
                dtype=torch.float).to(self.device)

            with torch.no_grad():
                ctr_pred, dur_pred = self.model(
                    u_feat_t.expand(B,-1), u_emb_t.expand(B,-1),
                    i_feat_b, i_emb_b,
                    seq_t.expand(B,-1), mask_t.expand(B,-1))

            final = alpha * ctr_pred + (1 - alpha) * dur_pred
            for iid, ctr, dur, fs in zip(
                b_iids,
                ctr_pred.cpu().numpy(),
                dur_pred.cpu().numpy(),
                final.cpu().numpy(),
            ):
                scores.append((iid, float(ctr), float(dur), float(fs)))

        # ── 阶段3: 排序取 Top N ──
        scores.sort(key=lambda x: x[3], reverse=True)
        return scores[:top_n]

    def batch_recommend(
        self,
        user_ids: List[int],
        recall_k: int   = 200,
        top_n:    int   = 10,
        alpha:    float = 0.5,
    ) -> Dict[int, List[Tuple]]:
        """批量推荐（多用户）"""
        return {
            uid: self.recommend(uid, recall_k, top_n, alpha)
            for uid in user_ids
        }
