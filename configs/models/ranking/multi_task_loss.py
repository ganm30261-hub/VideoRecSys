# models/ranking/multi_task_loss.py
# 多任务 Loss — KuaiRec 版本
# 对应 Step3 Cell 5
# ============================================================

import torch
import torch.nn as nn
from typing import Tuple


class MultiTaskLoss(nn.Module):
    """
    多任务 Loss — KuaiRec 版本

    任务1: CTR 预估（BCE Loss）
        标签: watch_ratio >= threshold → 1，否则 0

    任务2: 完播率预估（MSE Loss）
        标签: watch_ratio 归一化值（watch_ratio / watch_ratio_max）

    总 Loss:
        L = alpha * BCE(CTR) + (1-alpha) * MSE(完播率)

    KuaiRec 对应企业:
        快手/抖音用多目标 Loss 同时优化点击率和完播率
        alpha 是业务超参，完播率越重要 alpha 越小
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha: CTR loss 权重（0~1）
                   alpha=1.0 → 纯 CTR 优化
                   alpha=0.5 → CTR 和完播率各占一半
                   alpha=0.0 → 纯完播率优化
        """
        super().__init__()
        self.alpha    = alpha
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self,
        ctr_pred:   torch.Tensor,  # [B] CTR 预测概率
        dur_pred:   torch.Tensor,  # [B] 完播率预测值
        ctr_label:  torch.Tensor,  # [B] 二值 CTR 标签
        dur_label:  torch.Tensor,  # [B] 完播率归一化标签
    ) -> Tuple[torch.Tensor, float, float]:
        """
        Returns:
            total_loss: 总 Loss
            ctr_loss:   CTR Loss（用于监控）
            dur_loss:   完播率 Loss（用于监控）
        """
        l_ctr = self.bce_loss(ctr_pred, ctr_label)
        l_dur = self.mse_loss(dur_pred, dur_label)
        total = self.alpha * l_ctr + (1 - self.alpha) * l_dur
        return total, l_ctr.item(), l_dur.item()
