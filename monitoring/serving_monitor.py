# monitoring/serving_monitor.py
# 推荐链路性能监控 — KuaiRec 版本
# 对应 Step4 Cell 7
# ============================================================

import time
import logging
import numpy as np
import faiss
from typing import Dict, List

logger = logging.getLogger(__name__)


class ServingMonitor:
    """
    推荐链路性能监控

    监控项:
        召回延迟 (P50 / P90 / P99)
        精排延迟
        端到端延迟

    对应企业:
        Prometheus + Grafana 实时监控
        P99 > 50ms → 触发告警
    """

    def __init__(self):
        self._latency_records: Dict[str, List[float]] = {
            'recall'  : [],
            'ranking' : [],
            'e2e'     : [],
        }

    def benchmark_recall(
        self,
        faiss_index,
        user_emb_matrix: np.ndarray,
        user_id2row:     Dict,
        user_ids:        List,
        recall_k:        int = 200,
    ) -> Dict[str, float]:
        """
        双塔召回延迟压测

        Args:
            faiss_index:     FaissIndex 实例或原始 faiss index
            user_emb_matrix: 用户 Embedding 矩阵（L2归一化）
            user_id2row:     user_id → 矩阵行号
            user_ids:        压测用户列表
            recall_k:        召回候选数

        Returns:
            延迟统计字典 {p50, p90, p99, mean, n}
        """
        latencies = []
        for uid in user_ids:
            if uid not in user_id2row:
                continue
            t0    = time.perf_counter()
            u_vec = user_emb_matrix[user_id2row[uid]:user_id2row[uid]+1]
            if hasattr(faiss_index, 'search'):
                faiss_index.search(u_vec, recall_k)
            else:
                faiss_index.index.search(u_vec, recall_k)
            latencies.append((time.perf_counter() - t0) * 1000)

        self._latency_records['recall'].extend(latencies)
        return self._compute_stats(latencies, 'recall')

    def record_e2e(self, latency_ms: float) -> None:
        """记录端到端延迟"""
        self._latency_records['e2e'].append(latency_ms)

    def get_report(self) -> Dict[str, Dict]:
        """获取所有阶段的延迟报告"""
        report = {}
        for stage, lats in self._latency_records.items():
            if lats:
                report[stage] = self._compute_stats(lats, stage)
        return report

    def print_report(self, stats: Dict[str, float], stage: str = '') -> None:
        """打印延迟报告"""
        from tabulate import tabulate
        rows = [
            ['P50 延迟', f"{stats.get('p50', 0):.2f} ms", '< 10ms 达标'],
            ['P90 延迟', f"{stats.get('p90', 0):.2f} ms", '< 20ms 达标'],
            ['P99 延迟', f"{stats.get('p99', 0):.2f} ms", '< 50ms 达标'],
            ['均值延迟', f"{stats.get('mean', 0):.2f} ms", ''],
            ['测试样本', f"{stats.get('n', 0)}",           ''],
        ]
        title = f'【{stage} 性能测试】' if stage else '【链路性能测试】'
        print(f'\n{title}')
        print(tabulate(rows, headers=['指标', '数值', '参考标准'], tablefmt='grid'))

    def _compute_stats(self, latencies: List[float], stage: str) -> Dict:
        arr = np.array(latencies)
        stats = {
            'p50' : float(np.percentile(arr, 50)),
            'p90' : float(np.percentile(arr, 90)),
            'p99' : float(np.percentile(arr, 99)),
            'mean': float(np.mean(arr)),
            'n'   : len(arr),
        }
        logger.info(
            f"[{stage}] P50={stats['p50']:.2f}ms  "
            f"P90={stats['p90']:.2f}ms  "
            f"P99={stats['p99']:.2f}ms  "
            f"n={stats['n']}"
        )
        return stats
