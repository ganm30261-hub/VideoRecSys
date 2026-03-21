# monitoring/psi_monitor.py
# PSI 特征漂移监控 + 预测分布监控 — KuaiRec 版本
# 对应 Step4 Cell 5 + Cell 6
# ============================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

PSI_STABLE  = 0.1
PSI_WARNING = 0.2


class PSIMonitor:
    """
    PSI 特征漂移监控 — KuaiRec 版本

    KuaiRec 特有监控项:
        avg_watch_ratio  → 完播率分布（最核心业务指标）
        positive_rate    → 正样本率（watch_ratio >= 0.5 的比例）
        interaction_count → 用户活跃度
        watch_ratio（交互级）→ 观看行为整体分布

    PSI 阈值:
        < 0.1   → ✅ 稳定，无需处理
        0.1~0.2 → ⚠️  轻微漂移，关注
        >= 0.2  → 🚨 显著漂移，触发重训

    对应企业: 模型上线后每日/每周定时运行
    """

    def __init__(self, bins: int = 10):
        self.bins = bins

    def compute_psi(
        self,
        base: pd.Series,
        curr: pd.Series,
    ) -> float:
        """计算单个特征的 PSI"""
        b = base.dropna().values
        c = curr.dropna().values
        if len(b) == 0 or len(c) == 0:
            return 0.0

        bps = np.unique(np.percentile(b, np.linspace(0, 100, self.bins + 1)))
        if len(bps) < 2:
            return 0.0

        bc = np.histogram(b, bins=bps)[0]
        cc = np.histogram(c, bins=bps)[0]
        bp = np.where(bc == 0, 1e-6, bc / bc.sum())
        cp = np.where(cc == 0, 1e-6, cc / cc.sum())
        return float(np.sum((cp - bp) * np.log(cp / bp)))

    def monitor_features(
        self,
        base_df:  pd.DataFrame,
        curr_df:  pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """
        批量监控特征 PSI

        Args:
            base_df:  基准数据（训练集用户特征）
            curr_df:  当前数据（测试集/近期线上数据）
            features: 需要监控的特征列表

        Returns:
            PSI 报告 DataFrame
        """
        logger.info(
            f"特征 PSI 监控  基准={len(base_df):,}  当前={len(curr_df):,}"
        )
        rows, alerts = [], []

        for feat in features:
            if feat not in base_df.columns or feat not in curr_df.columns:
                logger.warning(f"  特征 {feat} 不存在，跳过")
                continue

            psi    = self.compute_psi(base_df[feat], curr_df[feat])
            status, action = self._evaluate(psi)
            rows.append({
                'feature': feat,
                'psi'    : round(psi, 4),
                'status' : status,
                'action' : action,
            })
            if psi >= PSI_WARNING:
                alerts.append(feat)
            logger.info(f"  {feat:35s} PSI={psi:.4f}  {status}")

        if alerts:
            logger.warning(f"🚨 告警特征: {alerts} → 触发模型重训")
        else:
            logger.info("✅ 所有特征分布稳定")

        return pd.DataFrame(rows)

    def monitor_watch_ratio(
        self,
        base_interactions: pd.DataFrame,
        curr_interactions: pd.DataFrame,
    ) -> float:
        """
        KuaiRec 特有：监控 watch_ratio 交互级分布
        对应企业: 监控用户整体观看行为是否发生漂移
        """
        if 'watch_ratio' not in base_interactions.columns:
            return 0.0
        psi    = self.compute_psi(
            base_interactions['watch_ratio'].clip(0, 3),
            curr_interactions['watch_ratio'].clip(0, 3),
        )
        status, _ = self._evaluate(psi)
        logger.info(f"  watch_ratio 分布 PSI={psi:.4f}  {status}")
        return psi

    def monitor_predictions(
        self,
        pred_t1: np.ndarray,
        pred_t2: np.ndarray,
        name:    str = '预测分布',
    ) -> float:
        """
        监控模型预测分布变化
        pred_t1: 模型上线初期的预测分数分布
        pred_t2: 近期的预测分数分布

        对应企业: 模型上线后每日监控预测分布
        """
        psi    = self.compute_psi(pd.Series(pred_t1), pd.Series(pred_t2))
        status, action = self._evaluate(psi)
        logger.info(f"  {name} PSI={psi:.4f}  {status}  → {action}")
        return psi

    def _evaluate(self, psi: float) -> Tuple[str, str]:
        if psi < PSI_STABLE:
            return '✅ 稳定',    '无需处理'
        elif psi < PSI_WARNING:
            return '⚠️  轻微漂移', '持续监控'
        else:
            return '🚨 显著漂移', '重新训练模型'
