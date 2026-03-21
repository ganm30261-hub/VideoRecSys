# experiment/metrics_calculator.py
# 实验指标计算 + 统计显著性检验 — KuaiRec 版本
# 对应 Step4 Cell 4
# ============================================================

import logging
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    实验指标计算器 — KuaiRec 版本

    KuaiRec 核心业务指标:
        ctr:       点击率（label=1 的比例）
        avg_watch: 平均完播率（watch_ratio 均值）← 最核心
        ndcg:      NDCG@10 排序质量

    对应企业:
        快手/抖音的核心指标是完播率，不是 CTR
        完播率提升才代表用户真实满意度提升
    """

    def compute_user_metrics(
        self,
        df:       pd.DataFrame,
        user_ids: List,
    ) -> Dict[int, Dict[str, float]]:
        """
        计算每个用户的指标

        Args:
            df:       交互数据（含 user_id, item_id, label, watch_ratio）
            user_ids: 需要计算的用户列表

        Returns:
            {user_id: {'ctr': float, 'avg_watch': float, 'ndcg': float}}
        """
        subset  = df[df['user_id'].isin(set(user_ids))]
        results = {}

        for uid, group in subset.groupby('user_id'):
            ctr = group['label'].mean()

            # KuaiRec 核心：平均完播率
            avg_watch = (group['watch_ratio'].clip(0, 3).mean()
                         if 'watch_ratio' in group.columns else ctr)

            # NDCG@10
            scores = (group['watch_ratio'].fillna(0).values
                      if 'watch_ratio' in group.columns
                      else group['label'].values)
            k      = min(10, len(group))
            ranked = (group.assign(score=scores)
                      .sort_values('score', ascending=False)
                      .head(k))
            dcg    = sum(row['label'] / np.log2(i + 2)
                         for i, (_, row) in enumerate(ranked.iterrows()))
            ideal  = sorted(group['label'].tolist(), reverse=True)[:k]
            idcg   = sum(v / np.log2(i + 2) for i, v in enumerate(ideal))

            results[uid] = {
                'ctr'      : float(ctr),
                'avg_watch': float(avg_watch),
                'ndcg'     : float(dcg / idcg) if idcg > 0 else 0.0,
            }

        return results


class StatisticalTester:
    """
    统计显著性检验器

    方法:
        双样本 T-test
        H0: 实验组 = 对照组（无差异）
        H1: 实验组 > 对照组（实验组更好）
        p < 0.05 → 拒绝 H0，差异统计显著，可以上线

    效果量:
        Cohen's d: 衡量差异的实际意义（不受样本量影响）
        |d| < 0.2  → 微小效果
        |d| < 0.5  → 小效果
        |d| >= 0.5 → 中等以上效果
    """

    def ttest(
        self,
        ctrl_metrics: Dict[int, Dict],
        trt_metrics:  Dict[int, Dict],
        metric:       str,
    ) -> Dict:
        """
        双样本 T-test

        Args:
            ctrl_metrics: 对照组用户指标
            trt_metrics:  实验组用户指标
            metric:       指标名称（'ctr', 'avg_watch', 'ndcg'）

        Returns:
            {ctrl_mean, trt_mean, lift, p_value, cohens_d, significant, ...}
        """
        ctrl_vals = [v[metric] for v in ctrl_metrics.values()]
        trt_vals  = [v[metric] for v in trt_metrics.values()]

        t_stat, p_value = stats.ttest_ind(trt_vals, ctrl_vals)
        ctrl_mean = np.mean(ctrl_vals)
        trt_mean  = np.mean(trt_vals)
        lift      = ((trt_mean - ctrl_mean) / ctrl_mean * 100
                     if ctrl_mean > 0 else 0.0)

        # Cohen's d 效果量
        pooled_std = np.sqrt(
            (np.std(ctrl_vals) ** 2 + np.std(trt_vals) ** 2) / 2)
        cohens_d   = ((trt_mean - ctrl_mean) / pooled_std
                      if pooled_std > 0 else 0.0)

        return {
            'metric'     : metric,
            'ctrl_mean'  : float(ctrl_mean),
            'trt_mean'   : float(trt_mean),
            'lift'       : float(lift),
            't_stat'     : float(t_stat),
            'p_value'    : float(p_value),
            'cohens_d'   : float(cohens_d),
            'significant': bool(p_value < 0.05),
            'ctrl_n'     : len(ctrl_vals),
            'trt_n'      : len(trt_vals),
        }

    def run_experiment_analysis(
        self,
        ctrl_metrics: Dict[int, Dict],
        trt_metrics:  Dict[int, Dict],
        metrics:      List[str],
        exp_name:     str,
        ctrl_desc:    str,
        trt_desc:     str,
    ) -> List[Dict]:
        """
        对一个实验的所有指标做统计检验，并打印结果

        Returns:
            每个指标的检验结果列表
        """
        from tabulate import tabulate

        results = []
        for metric in metrics:
            r = self.ttest(ctrl_metrics, trt_metrics, metric)
            results.append(r)

        print(f'\n【{exp_name}】  对照组={ctrl_desc}  实验组={trt_desc}')
        rows = [[
            r['metric'],
            f"{r['ctrl_mean']:.4f}",
            f"{r['trt_mean']:.4f}",
            f"{r['lift']:+.2f}%",
            f"{r['p_value']:.4f}",
            f"{r['cohens_d']:.3f}",
            '✅ 显著' if r['significant'] else '❌ 不显著',
        ] for r in results]
        print(tabulate(
            rows,
            headers=['指标', '对照组', '实验组', '提升', 'p值', "Cohen's d", '显著性'],
            tablefmt='grid'))

        return results
