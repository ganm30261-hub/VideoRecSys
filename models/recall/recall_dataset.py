# models/recall/recall_dataset.py
# 双塔训练 Dataset — KuaiRec 版本
# 对应 Step2 Cell 3
# ============================================================

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List


class RecallDataset(Dataset):
    """
    双塔召回训练 Dataset — KuaiRec 版本

    特点:
        - 只使用正样本（label=1），InfoNCE 在 batch 内做负采样
        - watch_ratio 作为样本权重（重复观看 = 强正样本）
        - 行为序列基于 pos_seq（正反馈历史）
    """

    def __init__(
        self,
        df,
        user_feat_dict:  Dict,
        item_feat_dict:  Dict,
        user2idx:        Dict,
        item2idx:        Dict,
        seq_dict:        Dict,
        seq_len:         int = 50,
        watch_ratio_max: float = 3.0,
        only_positive:   bool = True,
    ):
        """
        Args:
            df:               样本 DataFrame
            user_feat_dict:   {user_id: np.array}
            item_feat_dict:   {item_id: np.array}
            user2idx:         用户 ID → Embedding 索引
            item2idx:         视频 ID → Embedding 索引
            seq_dict:         {user_id: [item_idx, ...]} 正反馈序列
            seq_len:          序列最大长度
            watch_ratio_max:  watch_ratio 截断上限
            only_positive:    是否只保留正样本（双塔训练用 True）
        """
        if only_positive:
            self.df = df[df['label'] == 1].reset_index(drop=True)
        else:
            self.df = df.reset_index(drop=True)

        self.u_feat         = user_feat_dict
        self.i_feat         = item_feat_dict
        self.user2idx       = user2idx
        self.item2idx       = item2idx
        self.seq_dict       = seq_dict
        self.seq_len        = seq_len
        self.watch_ratio_max = watch_ratio_max

        self.u_dim  = len(next(iter(user_feat_dict.values())))
        self.i_dim  = len(next(iter(item_feat_dict.values())))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        uid    = row['user_id']
        iid    = row['item_id']
        weight = float(min(row.get('watch_ratio', 1.0), self.watch_ratio_max))

        u_idx  = self.user2idx.get(uid, 0)
        i_idx  = self.item2idx.get(iid, 0)
        u_vec  = self.u_feat.get(uid, np.zeros(self.u_dim,  np.float32))
        i_vec  = self.i_feat.get(iid, np.zeros(self.i_dim,  np.float32))

        # 行为序列（padding 到 seq_len）
        seq    = self.seq_dict.get(uid, [])
        seq    = seq[-self.seq_len:]
        pad    = self.seq_len - len(seq)
        seq_t  = torch.tensor([0] * pad + seq, dtype=torch.long)

        return (
            torch.tensor(u_idx,  dtype=torch.long),
            torch.tensor(i_idx,  dtype=torch.long),
            torch.tensor(u_vec,  dtype=torch.float),
            torch.tensor(i_vec,  dtype=torch.float),
            seq_t,
            torch.tensor(weight, dtype=torch.float),
        )
