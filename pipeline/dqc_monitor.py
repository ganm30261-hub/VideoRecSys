# pipeline/dqc_monitor.py
# 数据质量监控 + PSI 特征漂移检测 — KuaiRec 版本
# 对应 Step1 Cell 10
# ============================================================

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

PSI_STABLE  = 0.1
PSI_WARNING = 0.2


class DataQualityChecker:
    """
    数据质量检查 — KuaiRec 版本

    KuaiRec 特有检查:
        - watch_ratio 分布合理性（应在 0~5 之间）
        - 正样本率检查（KuaiRec 密度高，正样本率通常 > 50%）
        - 重复观看比例（watch_ratio > 1 的比例）
    """

    def run(self, df: pd.DataFrame) -> Dict:
        logger.info("开始数据质量检查（KuaiRec）...")
        report = {}
        report['basic']       = self._basic_stats(df)
        report['watch_ratio'] = self._watch_ratio_check(df)
        report['label']       = self._label_check(df)
        report['temporal']    = self._temporal_check(df)
        report['duplicate']   = self._duplicate_check(df)
        self._print_report(report)
        return report

    def _basic_stats(self, df: pd.DataFrame) -> Dict:
        return {
            'total_rows'  : len(df),
            'user_count'  : df['user_id'].nunique(),
            'item_count'  : df['item_id'].nunique(),
            'missing_rate': df.isnull().mean().to_dict(),
        }

    def _watch_ratio_check(self, df: pd.DataFrame) -> Dict:
        """KuaiRec 特有: watch_ratio 分布检查"""
        if 'watch_ratio' not in df.columns:
            return {}
        wr = df['watch_ratio']
        return {
            'mean'           : wr.mean(),
            'std'            : wr.std(),
            'pct_below_0'    : (wr < 0).mean(),       # 异常：负值
            'pct_above_5'    : (wr > 5).mean(),        # 异常：超出截断值
            'pct_rewatched'  : (wr > 1).mean(),        # 重复观看比例
            'pct_completed'  : (wr >= 1).mean(),       # 看完比例
        }

    def _label_check(self, df: pd.DataFrame) -> Dict:
        if 'label' not in df.columns:
            return {}
        pos_rate = df['label'].mean()
        # KuaiRec 正样本率通常较高（50%~85%）
        status = 'ok'
        if pos_rate < 0.1:
            status = 'warning: 正样本率过低，检查 watch_ratio_threshold'
        elif pos_rate > 0.95:
            status = 'warning: 正样本率过高，建议提高阈值至 1.0'
        return {
            'positive_rate': pos_rate,
            'negative_rate': 1 - pos_rate,
            'status'       : status,
        }

    def _temporal_check(self, df: pd.DataFrame) -> Dict:
        if 'timestamp_unix' not in df.columns:
            return {}
        ts = df['timestamp_unix']
        return {
            'ts_min'   : int(ts.min()),
            'ts_max'   : int(ts.max()),
            'span_days': (ts.max() - ts.min()) / 86400,
        }

    def _duplicate_check(self, df: pd.DataFrame) -> Dict:
        dup = df.duplicated(subset=['user_id', 'item_id']).sum()
        return {
            'duplicate_count': int(dup),
            'duplicate_rate' : dup / len(df),
        }

    def _print_report(self, report: Dict) -> None:
        basic = report.get('basic', {})
        label = report.get('label', {})
        wr    = report.get('watch_ratio', {})
        dup   = report.get('duplicate', {})

        logger.info("=== KuaiRec 数据质量报告 ===")
        logger.info(f"  总行数:       {basic.get('total_rows', 0):,}")
        logger.info(f"  用户数:       {basic.get('user_count', 0):,}")
        logger.info(f"  视频数:       {basic.get('item_count', 0):,}")
        logger.info(f"  正样本率:     {label.get('positive_rate', 0):.3f}  "
                    f"状态: {label.get('status', 'N/A')}")
        logger.info(f"  重复观看率:   {wr.get('pct_rewatched', 0):.3f}  "
                    f"(watch_ratio > 1)")
        logger.info(f"  完播率:       {wr.get('pct_completed', 0):.3f}  "
                    f"(watch_ratio >= 1)")
        logger.info(f"  重复记录率:   {dup.get('duplicate_rate', 0):.4f}")


class PSIMonitor:
    """
    PSI 特征漂移监控 — KuaiRec 版本

    KuaiRec 特有监控指标:
        avg_watch_ratio  → 完播率分布（最核心）
        positive_rate    → 正样本率
        interaction_count → 用户活跃度
        watch_ratio（交互级别）→ 观看行为分布

    PSI 阈值:
        < 0.1   → ✅ 稳定
        0.1~0.2 → ⚠️  轻微漂移，持续关注
        >= 0.2  → 🚨 显著漂移，需重新训练
    """

    def __init__(self, bins: int = 10):
        self.bins = bins

    def compute_psi(
        self,
        base_series: pd.Series,
        curr_series: pd.Series,
    ) -> float:
        base = base_series.dropna().values
        curr = curr_series.dropna().values
        if len(base) == 0 or len(curr) == 0:
            return 0.0

        breakpoints = np.unique(
            np.percentile(base, np.linspace(0, 100, self.bins + 1)))
        if len(breakpoints) < 2:
            return 0.0

        base_cnt = np.histogram(base, bins=breakpoints)[0]
        curr_cnt = np.histogram(curr, bins=breakpoints)[0]
        base_pct = np.where(base_cnt == 0, 1e-6, base_cnt / base_cnt.sum())
        curr_pct = np.where(curr_cnt == 0, 1e-6, curr_cnt / curr_cnt.sum())

        return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))

    def monitor(
        self,
        base_df: pd.DataFrame,
        curr_df: pd.DataFrame,
        features: List[str],
    ) -> pd.DataFrame:
        """
        批量监控多个特征的 PSI

        Args:
            base_df:  基准数据（训练集）
            curr_df:  当前数据（测试集或近期线上数据）
            features: 监控的特征列名列表

        Returns:
            PSI 报告 DataFrame
        """
        logger.info(
            f"PSI 监控  特征数={len(features)}  "
            f"基准={len(base_df):,}  当前={len(curr_df):,}"
        )
        rows, alerts = [], []

        for feat in features:
            if feat not in base_df.columns or feat not in curr_df.columns:
                logger.warning(f"  特征 {feat} 不存在，跳过")
                continue

            psi    = self.compute_psi(base_df[feat], curr_df[feat])
            status, action = self._evaluate(psi)
            rows.append({'feature': feat, 'psi': round(psi, 4),
                         'status': status, 'action': action})

            if psi >= PSI_WARNING:
                alerts.append(feat)
            logger.info(f"  {feat:35s} PSI={psi:.4f}  {status}")

        report = pd.DataFrame(rows)

        if alerts:
            logger.warning(f"🚨 告警特征: {alerts} → 建议重新训练模型")
        else:
            logger.info("✅ 所有特征分布稳定，无需重训")

        return report

    def monitor_watch_ratio(
        self,
        base_interactions: pd.DataFrame,
        curr_interactions: pd.DataFrame,
    ) -> float:
        """
        KuaiRec 特有: 监控 watch_ratio 交互级别分布
        对应企业: 监控用户观看行为是否发生整体漂移
        """
        if 'watch_ratio' not in base_interactions.columns:
            return 0.0
        psi = self.compute_psi(
            base_interactions['watch_ratio'].clip(0, 3),
            curr_interactions['watch_ratio'].clip(0, 3),
        )
        status, _ = self._evaluate(psi)
        logger.info(f"  watch_ratio 分布 PSI={psi:.4f}  {status}")
        return psi

    def _evaluate(self, psi: float) -> Tuple[str, str]:
        if psi < PSI_STABLE:
            return '✅ 稳定', '无需处理'
        elif psi < PSI_WARNING:
            return '⚠️  轻微漂移', '持续监控'
        else:
            return '🚨 显著漂移', '重新训练模型'


class SampleDistributionChecker:
    """
    样本分布检查 — KuaiRec 版本

    KuaiRec 特有检查:
        - watch_ratio 分布一致性（训练 vs 测试）
        - 完播率分布（pct_completed）一致性
    """

    def check(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        report = {}

        report['label'] = {
            'train_pos_rate': train_df['label'].mean(),
            'test_pos_rate' : test_df['label'].mean(),
        }

        # KuaiRec 特有：watch_ratio 分布
        if 'watch_ratio' in train_df.columns:
            report['watch_ratio'] = {
                'train_mean': train_df['watch_ratio'].mean(),
                'test_mean' : test_df['watch_ratio'].mean(),
                'train_completed': (train_df['watch_ratio'] >= 1).mean(),
                'test_completed' : (test_df['watch_ratio'] >= 1).mean(),
            }

        # 用户覆盖
        train_users = set(train_df['user_id'].unique())
        test_users  = set(test_df['user_id'].unique())
        report['user_coverage'] = {
            'train_users'   : len(train_users),
            'test_users'    : len(test_users),
            'overlap_rate'  : len(train_users & test_users) / len(test_users),
        }

        # 时序不穿越验证
        if 'timestamp_unix' in train_df.columns:
            train_max = train_df['timestamp_unix'].quantile(0.95)
            test_min  = test_df['timestamp_unix'].quantile(0.05)
            report['temporal'] = {
                'no_leakage': test_min >= train_max,
            }

        self._print_report(report)
        return report

    def _print_report(self, report: Dict) -> None:
        logger.info("=== 样本分布检查（KuaiRec）===")
        label = report.get('label', {})
        logger.info(f"  训练集正样本率: {label.get('train_pos_rate', 0):.3f}")
        logger.info(f"  测试集正样本率: {label.get('test_pos_rate',  0):.3f}")

        wr = report.get('watch_ratio', {})
        if wr:
            logger.info(f"  训练集平均完播率: {wr.get('train_mean', 0):.3f}")
            logger.info(f"  测试集平均完播率: {wr.get('test_mean',  0):.3f}")

        temporal = report.get('temporal', {})
        if temporal:
            flag = '✅' if temporal.get('no_leakage') else '🚨 样本穿越！'
            logger.info(f"  时序一致性: {flag}")
