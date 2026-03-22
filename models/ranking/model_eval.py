# models/ranking/model_eval.py
# 精排离线评估 — KuaiRec 版本
# 对应 Step3 Cell 7
# ============================================================

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, mean_squared_error
from typing import Dict, List

logger = logging.getLogger(__name__)


class RankingEvaluator:
    """
    精排离线评估器 — KuaiRec 版本

    评估指标:
        CTR 任务:
            AUC      → 整体排序能力（ROC曲线面积）
            GAUC     → 分用户 AUC 均值（更公平，工业标准）
            LogLoss  → CTR 预测校准度
        排序质量:
            NDCG@K   → 归一化折损累积增益
        完播率任务（KuaiRec 特有）:
            RMSE     → 完播率预测均方根误差
    """

    def evaluate(
        self,
        test_df:    pd.DataFrame,
        ctr_preds:  List[float],
        dur_preds:  List[float],
        ctr_labels: List[float],
        dur_labels: List[float],
        ks:         List[int] = [10, 20],
    ) -> Dict[str, float]:
        """
        Args:
            test_df:    测试集 DataFrame（含 user_id）
            ctr_preds:  CTR 预测概率列表
            dur_preds:  完播率预测值列表
            ctr_labels: CTR 真实标签列表
            dur_labels: 完播率真实标签列表
            ks:         NDCG 的 K 值列表

        Returns:
            评估指标字典
        """
        results = {}

        # CTR 指标
        results['AUC']     = roc_auc_score(ctr_labels, ctr_preds)
        results['LogLoss'] = log_loss(ctr_labels, ctr_preds)
        results['GAUC']    = self._compute_gauc(
            test_df, ctr_preds, ctr_labels)

        # NDCG
        for k in ks:
            results[f'NDCG@{k}'] = self._compute_ndcg(
                test_df, ctr_preds, k=k)

        # 完播率 RMSE（KuaiRec 特有）
        results['WatchRatio_RMSE'] = float(
            np.sqrt(mean_squared_error(dur_labels, dur_preds)))

        return results

    def print_report(self, results: Dict[str, float]) -> None:
        """打印评估报告"""
        from tabulate import tabulate
        ref_map = {
            'AUC'              : '>0.72 达标',
            'GAUC'             : '>0.68 达标',
            'LogLoss'          : '<0.50 达标',
            'NDCG@10'          : '越高越好',
            'NDCG@20'          : '越高越好',
            'WatchRatio_RMSE'  : '越低越好',
        }
        rows = [
            [k, f'{v:.4f}', ref_map.get(k, '')]
            for k, v in results.items()
        ]
        print('\n【DIN 多任务精排离线评估】')
        print(tabulate(rows, headers=['指标', '数值', '参考标准'], tablefmt='grid'))

    # ─────────────────────────────────────────
    # 私有方法
    # ─────────────────────────────────────────

    def _compute_gauc(
        self,
        df:     pd.DataFrame,
        preds:  List[float],
        labels: List[float],
    ) -> float:
        """
        GAUC: 分用户计算 AUC 后按曝光数加权平均
        比整体 AUC 更能反映个性化排序能力
        """
        df = df.copy()
        df['pred']  = preds
        df['label'] = labels
        gauc_sum, weight_sum = 0.0, 0

        for uid, group in df.groupby('user_id'):
            if group['label'].nunique() < 2:
                continue
            auc = roc_auc_score(group['label'], group['pred'])
            gauc_sum   += auc * len(group)
            weight_sum += len(group)

        return gauc_sum / weight_sum if weight_sum > 0 else 0.0

    def _compute_ndcg(
        self,
        df:    pd.DataFrame,
        preds: List[float],
        k:     int = 10,
    ) -> float:
        """NDCG@K: 分用户计算后平均"""
        df = df.copy()
        df['pred'] = preds
        ndcg_list  = []

        for uid, group in df.groupby('user_id'):
            if group['label'].sum() == 0:
                continue
            group = group.sort_values('pred', ascending=False).head(k)
            dcg   = sum(
                row['label'] / np.log2(i + 2)
                for i, (_, row) in enumerate(group.iterrows()))
            idcg  = sum(
                1 / np.log2(i + 2)
                for i in range(min(int(group['label'].sum()), k)))
            ndcg_list.append(dcg / idcg if idcg > 0 else 0)

        return float(np.mean(ndcg_list)) if ndcg_list else 0.0
