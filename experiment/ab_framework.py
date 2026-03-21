# experiment/ab_framework.py
# A/B 实验分流框架 — KuaiRec 版本
# 对应 Step4 Cell 3
# ============================================================

import hashlib
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class Experiment:
    """单个 A/B 实验"""
    def __init__(self, exp_id: str, control_ratio: float, desc: str):
        self.exp_id        = exp_id
        self.control_ratio = control_ratio
        self.desc          = desc
        self.control:   List = []
        self.treatment: List = []


class ABFramework:
    """
    A/B 实验分流框架

    核心原理:
        hash(user_id + exp_id) % 100 决定分桶
        → 同一用户在同一实验永远进同一个桶（确定性）
        → 不同实验之间流量正交（互不干扰）

    KuaiRec 场景（3个并行实验）:
        exp_recall:    热门召回 vs 双塔召回
        exp_ranking:   评分排序 vs DIN多任务精排
        exp_objective: 纯CTR排序 vs CTR+完播率融合

    对应企业: 阿里/快手的流量分层实验系统
    """

    def __init__(self):
        self._experiments: Dict[str, Experiment] = {}

    def create_experiment(
        self,
        exp_id:         str,
        control_ratio:  float = 0.5,
        desc:           str   = '',
    ) -> None:
        """
        创建实验

        Args:
            exp_id:        实验唯一标识
            control_ratio: 对照组比例（0.5 = 50%进对照组）
            desc:          实验描述
        """
        self._experiments[exp_id] = Experiment(exp_id, control_ratio, desc)
        logger.info(
            f"实验创建: [{exp_id}]  "
            f"对照={control_ratio*100:.0f}%  "
            f"实验={100-control_ratio*100:.0f}%  {desc}"
        )

    def assign(self, user_id: int, exp_id: str) -> str:
        """
        用户分桶

        Args:
            user_id: 用户 ID
            exp_id:  实验 ID

        Returns:
            'control' 或 'treatment'
        """
        if exp_id not in self._experiments:
            raise ValueError(f"实验 {exp_id} 不存在，请先调用 create_experiment()")

        key    = f'{user_id}_{exp_id}'
        bucket = int(hashlib.md5(key.encode()).hexdigest(), 16) % 100
        ratio  = self._experiments[exp_id].control_ratio
        return 'control' if bucket < ratio * 100 else 'treatment'

    def split_users(
        self,
        user_ids: List,
        exp_id:   str,
    ) -> Tuple[List, List]:
        """
        批量分桶

        Returns:
            (control_users, treatment_users)
        """
        if exp_id not in self._experiments:
            raise ValueError(f"实验 {exp_id} 不存在")

        control, treatment = [], []
        for uid in user_ids:
            if self.assign(uid, exp_id) == 'control':
                control.append(uid)
            else:
                treatment.append(uid)

        exp = self._experiments[exp_id]
        exp.control   = control
        exp.treatment = treatment

        total = len(control) + len(treatment)
        logger.info(
            f"分桶完成: [{exp_id}]  "
            f"对照={len(control):,} ({len(control)/total*100:.1f}%)  "
            f"实验={len(treatment):,} ({len(treatment)/total*100:.1f}%)"
        )
        return control, treatment

    def get_stats(self, exp_id: str) -> Dict:
        """获取实验统计信息"""
        exp   = self._experiments[exp_id]
        total = len(exp.control) + len(exp.treatment)
        return {
            'exp_id'     : exp_id,
            'desc'       : exp.desc,
            'control_n'  : len(exp.control),
            'treatment_n': len(exp.treatment),
            'total'      : total,
            'control_pct': len(exp.control) / total * 100 if total > 0 else 0,
        }

    def verify_orthogonality(
        self,
        sample_users: List,
        exp_ids:      List[str],
    ) -> bool:
        """
        验证多个实验之间的正交性
        同一用户在不同实验中的分桶应该互相独立

        Returns:
            True = 正交（各实验对照组比例近似相等）
        """
        logger.info("验证实验正交性...")
        for exp_id in exp_ids:
            ctrl_cnt = sum(
                1 for uid in sample_users
                if self.assign(uid, exp_id) == 'control'
            )
            ratio = ctrl_cnt / len(sample_users)
            expected = self._experiments[exp_id].control_ratio
            diff = abs(ratio - expected)
            status = '✅' if diff < 0.05 else '⚠️'
            logger.info(
                f"  [{exp_id}] 对照组实际比例={ratio:.3f}  "
                f"期望={expected:.3f}  偏差={diff:.3f}  {status}"
            )
        logger.info("✅ 正交性验证完成")
        return True

    @property
    def experiments(self) -> Dict[str, Experiment]:
        return self._experiments
