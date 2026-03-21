# models/ranking/ranking_dataset.py
# 精排训练 Dataset — KuaiRec 版本
# 对应 Step3 Cell 3
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict


class RankingDataset(Dataset):
    """
    精排训练 Dataset — KuaiRec 版本

    特点:
        正负样本都用（不同于召回只用正样本）
        多任务标签:
            label:       二值 CTR（watch_ratio >= threshold）
            watch_ratio: 连续值，归一化后作为完播率回归目标
    """

    def __init__(
        self,
        df,
        user_feat_dict:   Dict,
        item_feat_dict:   Dict,
        user_emb_dict:    Dict,
        item_emb_dict:    Dict,
        seq_dict:         Dict,
        item2idx:         Dict,
        seq_len:          int   = 50,
        watch_ratio_max:  float = 3.0,
    ):
        self.df              = df.reset_index(drop=True)
        self.u_feat          = user_feat_dict
        self.i_feat          = item_feat_dict
        self.u_emb           = user_emb_dict
        self.i_emb           = item_emb_dict
        self.seq_dict        = seq_dict
        self.item2idx        = item2idx
        self.seq_len         = seq_len
        self.watch_ratio_max = watch_ratio_max

        self.u_dim   = len(next(iter(user_feat_dict.values())))
        self.i_dim   = len(next(iter(item_feat_dict.values())))
        self.emb_dim = len(next(iter(user_emb_dict.values())))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        uid    = row['user_id']
        iid    = row['item_id']

        # 多任务标签
        ctr_label = float(row['label'])
        watch_raw = float(row.get('watch_ratio', 0.0))
        # 归一化到 [0, 1]
        dur_label = min(watch_raw, self.watch_ratio_max) / self.watch_ratio_max

        # 特征
        u_vec = self.u_feat.get(uid, np.zeros(self.u_dim,  np.float32))
        u_emb = self.u_emb.get(uid, np.zeros(self.emb_dim, np.float32))
        i_vec = self.i_feat.get(iid, np.zeros(self.i_dim,  np.float32))
        i_emb = self.i_emb.get(iid, np.zeros(self.emb_dim, np.float32))

        # 行为序列
        seq      = self.seq_dict.get(uid, [])
        seq      = seq[-self.seq_len:]
        pad_len  = self.seq_len - len(seq)
        seq_pad  = torch.tensor([0] * pad_len + seq, dtype=torch.long)
        seq_mask = torch.tensor([0] * pad_len + [1] * len(seq), dtype=torch.float)

        return (
            torch.tensor(u_vec,     dtype=torch.float),
            torch.tensor(u_emb,     dtype=torch.float),
            torch.tensor(i_vec,     dtype=torch.float),
            torch.tensor(i_emb,     dtype=torch.float),
            seq_pad,
            seq_mask,
            torch.tensor(ctr_label, dtype=torch.float),
            torch.tensor(dur_label, dtype=torch.float),
        )
