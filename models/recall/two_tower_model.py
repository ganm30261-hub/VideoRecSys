# models/recall/two_tower_model.py
# 双塔模型定义 — KuaiRec 版本
# 对应 Step2 Cell 4
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_dim: int, hidden_layers: list, dropout: float) -> nn.Sequential:
    """构建 MLP 模块（含 BN + ReLU + Dropout）"""
    layers = []
    dims   = [input_dim] + hidden_layers
    for i in range(len(dims) - 1):
        layers += [
            nn.Linear(dims[i], dims[i+1]),
            nn.BatchNorm1d(dims[i+1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
    return nn.Sequential(*layers)


class UserTower(nn.Module):
    """
    用户塔
    输入: 用户ID Embedding + 数值特征 + 行为序列（Mean Pooling）
    输出: L2 归一化的用户向量

    KuaiRec 特有:
        行为序列基于 watch_ratio >= 0.5 的正反馈序列
        Mean Pooling 时忽略 padding 位置（mask）
    """

    def __init__(self, n_users: int, u_dim: int, n_items: int, cfg: dict):
        super().__init__()
        emb_dim   = cfg['embedding_dim']
        self.user_emb = nn.Embedding(n_users + 1, emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        input_dim = emb_dim + u_dim + emb_dim   # ID emb + 特征 + 序列emb
        self.mlp  = build_mlp(input_dim, cfg['hidden_layers'], cfg['dropout'])
        self.out  = nn.Linear(cfg['hidden_layers'][-1], emb_dim)

    def forward(
        self,
        user_idx: torch.Tensor,   # [B]
        u_feat:   torch.Tensor,   # [B, u_dim]
        seq_idx:  torch.Tensor,   # [B, L]
    ) -> torch.Tensor:            # [B, emb_dim] L2归一化
        uid_emb = self.user_emb(user_idx)              # [B, D]
        seq_emb = self.item_emb(seq_idx)               # [B, L, D]
        # Mask padding（idx=0），只对有效位置取均值
        mask    = (seq_idx > 0).float().unsqueeze(-1)  # [B, L, 1]
        seq_agg = (seq_emb * mask).sum(1) / (mask.sum(1) + 1e-8)  # [B, D]
        x = torch.cat([uid_emb, u_feat, seq_agg], dim=-1)
        x = self.out(self.mlp(x))
        return F.normalize(x, dim=-1)


class ItemTower(nn.Module):
    """
    视频塔
    输入: 视频ID Embedding + 数值特征（含 KuaiRec 类别 + 每日统计特征）
    输出: L2 归一化的视频向量
    """

    def __init__(self, n_items: int, i_dim: int, cfg: dict):
        super().__init__()
        emb_dim   = cfg['embedding_dim']
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        input_dim = emb_dim + i_dim
        self.mlp  = build_mlp(input_dim, cfg['hidden_layers'], cfg['dropout'])
        self.out  = nn.Linear(cfg['hidden_layers'][-1], emb_dim)

    def forward(
        self,
        item_idx: torch.Tensor,  # [B]
        i_feat:   torch.Tensor,  # [B, i_dim]
    ) -> torch.Tensor:           # [B, emb_dim] L2归一化
        iid_emb = self.item_emb(item_idx)
        x = torch.cat([iid_emb, i_feat], dim=-1)
        x = self.out(self.mlp(x))
        return F.normalize(x, dim=-1)


class TwoTowerModel(nn.Module):
    """
    双塔模型封装（方便整体 save/load）
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        u_dim:   int,
        i_dim:   int,
        cfg:     dict,
    ):
        super().__init__()
        self.user_tower = UserTower(n_users, u_dim, n_items, cfg)
        self.item_tower = ItemTower(n_items, i_dim, cfg)
        self.cfg        = cfg

    def forward(self, user_idx, u_feat, seq_idx, item_idx, i_feat):
        u_emb = self.user_tower(user_idx, u_feat, seq_idx)
        i_emb = self.item_tower(item_idx, i_feat)
        return u_emb, i_emb

    def encode_user(self, user_idx, u_feat, seq_idx):
        return self.user_tower(user_idx, u_feat, seq_idx)

    def encode_item(self, item_idx, i_feat):
        return self.item_tower(item_idx, i_feat)
