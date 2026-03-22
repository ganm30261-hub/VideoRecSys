# models/recall/weighted_infonce_loss.py
# 加权 InfoNCE Loss — KuaiRec 版本
# 对应 Step2 Cell 5
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class WeightedInfoNCELoss(nn.Module):
    """
    加权 InfoNCE Loss（In-batch Negative Sampling）

    标准 InfoNCE:
        L = -log( exp(sim(u,i+)/τ) / Σj exp(sim(u,ij)/τ) )

    KuaiRec 加权版本:
        L = -w * log( exp(sim(u,i+)/τ) / Σj exp(sim(u,ij)/τ) )
        w = watch_ratio（重复观看的强正样本权重更高）

    对应企业: 快手/抖音用完播率对训练样本加权，
              让模型更关注真正受欢迎的视频
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        user_emb:  torch.Tensor,           # [B, D] L2归一化
        item_emb:  torch.Tensor,           # [B, D] L2归一化
        weights:   Optional[torch.Tensor] = None,  # [B] watch_ratio 权重
    ) -> torch.Tensor:
        # 相似度矩阵 [B, B]
        sim    = torch.matmul(user_emb, item_emb.T) / self.temperature
        labels = torch.arange(sim.size(0), device=sim.device)

        # 双向 loss（用户→视频 + 视频→用户）
        loss_u2i = F.cross_entropy(sim,   labels, reduction='none')  # [B]
        loss_i2u = F.cross_entropy(sim.T, labels, reduction='none')  # [B]
        loss     = (loss_u2i + loss_i2u) / 2                         # [B]

        # watch_ratio 加权
        if weights is not None:
            # 归一化防止梯度爆炸
            weights = weights / (weights.mean() + 1e-8)
            loss    = (loss * weights).mean()
        else:
            loss = loss.mean()

        return loss
