# models/ranking/din_model.py
# DIN 多任务精排模型 — KuaiRec 版本
# 对应 Step3 Cell 4
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINAttention(nn.Module):
    """
    DIN 注意力单元
    计算目标视频与历史行为序列中每个视频的相关性权重

    输入拼接方式 (DIN 原论文):
        [目标emb, 序列emb, 差值, 点积] → 4D 拼接 → MLP → 注意力分数
    """

    def __init__(self, emb_dim: int, hidden: list = [64, 16]):
        super().__init__()
        dims   = [emb_dim * 4] + hidden + [1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        target_emb: torch.Tensor,  # [B, D]
        seq_emb:    torch.Tensor,  # [B, L, D]
        mask:       torch.Tensor,  # [B, L] 1=有效 0=padding
    ) -> torch.Tensor:             # [B, D] 加权序列表示
        B, L, D = seq_emb.shape
        t_exp   = target_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]

        # 4种交互特征拼接
        concat  = torch.cat([
            t_exp,
            seq_emb,
            t_exp - seq_emb,
            t_exp * seq_emb,
        ], dim=-1)  # [B, L, 4D]

        scores  = self.net(concat).squeeze(-1)        # [B, L]
        scores  = scores - (1 - mask) * 1e9           # mask padding
        weights = torch.softmax(scores, dim=-1)        # [B, L]
        return (weights.unsqueeze(-1) * seq_emb).sum(dim=1)  # [B, D]


class DINMultiTask(nn.Module):
    """
    多任务 DIN 精排模型 — KuaiRec 版本

    架构:
        共享底层 DNN（用户特征 + 视频特征 + 注意力输出）
        ├── CTR 预估头（sigmoid，二分类）
        └── 完播率预估头（sigmoid，回归 watch_ratio/max）

    KuaiRec 特有:
        同时预测 CTR 和完播率（watch_ratio）
        最终排序 = α * CTR + (1-α) * 完播率
        对应企业: 快手/抖音的多目标排序框架

    输入:
        u_feat:   用户数值特征 [B, u_dim]
        u_emb:    Step2 用户 Embedding [B, tower_emb_dim]
        i_feat:   视频数值特征 [B, i_dim]
        i_emb:    Step2 视频 Embedding [B, tower_emb_dim]
        seq_idx:  行为序列 item 索引 [B, L]
        seq_mask: 序列有效位 mask [B, L]

    输出:
        ctr_pred:      CTR 预测概率 [B]
        duration_pred: 完播率预测值 [B]
    """

    def __init__(
        self,
        n_items:       int,
        i_feat_dim:    int,
        u_feat_dim:    int,
        tower_emb_dim: int,
        cfg:           dict,
    ):
        super().__init__()
        emb_dim    = cfg['embedding_dim']
        att_hidden = cfg['attention_hidden']
        dnn_hidden = cfg['dnn_hidden']
        dropout    = cfg['dropout']

        # 序列行为 Embedding（独立学习，不复用 Step2）
        self.item_emb  = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)

        # DIN 注意力
        self.attention = DINAttention(emb_dim, att_hidden)

        # DNN 输入维度
        # = 用户特征 + 用户tower_emb + 视频特征 + 视频tower_emb + 注意力输出
        dnn_input_dim = (u_feat_dim + tower_emb_dim +
                         i_feat_dim + tower_emb_dim + emb_dim)

        # 共享底层 DNN
        shared_layers = []
        dims = [dnn_input_dim] + dnn_hidden[:-1]
        for i in range(len(dims) - 1):
            shared_layers += [
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
        self.shared_dnn = nn.Sequential(*shared_layers)

        # 独立输出 Tower
        last_dim  = dnn_hidden[-2] if len(dnn_hidden) > 1 else dnn_input_dim
        tower_dim = dnn_hidden[-1]

        self.ctr_tower = nn.Sequential(
            nn.Linear(last_dim, tower_dim),
            nn.ReLU(),
            nn.Linear(tower_dim, 1),
        )
        self.duration_tower = nn.Sequential(
            nn.Linear(last_dim, tower_dim),
            nn.ReLU(),
            nn.Linear(tower_dim, 1),
        )

    def forward(
        self,
        u_feat:   torch.Tensor,  # [B, u_dim]
        u_emb:    torch.Tensor,  # [B, tower_emb_dim]
        i_feat:   torch.Tensor,  # [B, i_dim]
        i_emb:    torch.Tensor,  # [B, tower_emb_dim]
        seq_idx:  torch.Tensor,  # [B, L]
        seq_mask: torch.Tensor,  # [B, L]
    ):
        # DIN 注意力：用视频 Embedding 前 emb_dim 维作为目标向量
        seq_emb  = self.item_emb(seq_idx)                       # [B, L, emb_dim]
        att_out  = self.attention(
            i_emb[:, :self.item_emb.embedding_dim],
            seq_emb, seq_mask)                                   # [B, emb_dim]

        # 拼接所有特征
        x = torch.cat([u_feat, u_emb, i_feat, i_emb, att_out], dim=-1)
        shared = self.shared_dnn(x)

        ctr_logit      = self.ctr_tower(shared).squeeze(-1)     # [B]
        duration_logit = self.duration_tower(shared).squeeze(-1) # [B]

        return torch.sigmoid(ctr_logit), torch.sigmoid(duration_logit)

    def predict_score(
        self,
        u_feat:   torch.Tensor,
        u_emb:    torch.Tensor,
        i_feat:   torch.Tensor,
        i_emb:    torch.Tensor,
        seq_idx:  torch.Tensor,
        seq_mask: torch.Tensor,
        alpha:    float = 0.5,
    ) -> torch.Tensor:
        """
        融合排序分数
        score = alpha * CTR + (1-alpha) * 完播率
        """
        ctr_pred, dur_pred = self.forward(
            u_feat, u_emb, i_feat, i_emb, seq_idx, seq_mask)
        return alpha * ctr_pred + (1 - alpha) * dur_pred
