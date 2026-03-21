# pipeline/data_cleaning.py
# 数据清洗模块 — KuaiRec 版本
# 对应 Step1 Cell 4
# ============================================================

import logging
import pandas as pd
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)


class DataCleaningPipeline:
    """
    KuaiRec 数据清洗 Pipeline

    KuaiRec 与 MovieLens 的核心差异:
        - 交互字段: watch_ratio（完播率）而非 rating（评分）
        - watch_ratio > 1 表示重复观看，是强正样本，需保留
        - small_matrix 可能无时间戳，自动生成伪时序
        - 密度 99.6%，无需担心稀疏性

    处理流程:
        1. 列名统一（video_id → item_id）
        2. watch_ratio 异常处理（clip + fillna）
        3. 添加二值标签
        4. 去重（保留 watch_ratio 最大的一条）
        5. 时间戳处理（无时间戳时生成伪时序）
        6. 冷启动过滤
        7. 时序排序
    """

    def __init__(
        self,
        watch_ratio_threshold: float = 0.5,
        watch_ratio_clip: float = 5.0,
        min_user_interactions: int = 10,
        min_item_interactions: int = 5,
    ):
        """
        Args:
            watch_ratio_threshold: 正样本阈值，>= 此值为正样本（默认0.5）
            watch_ratio_clip:      watch_ratio 截断上限（默认5.0）
            min_user_interactions: 冷启动过滤，用户最少交互数
            min_item_interactions: 冷启动过滤，视频最少被交互数
        """
        self.threshold   = watch_ratio_threshold
        self.clip_val    = watch_ratio_clip
        self.min_user    = min_user_interactions
        self.min_item    = min_item_interactions

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        执行完整清洗流程

        Args:
            df: 原始交互数据（small_matrix.csv 或 big_matrix.csv）

        Returns:
            清洗后的 DataFrame，包含列:
                user_id, item_id, watch_ratio, label, timestamp_unix
        """
        raw_count = len(df)
        logger.info(f"原始数据: {raw_count:,} 行")

        df = self._unify_columns(df)
        df = self._process_watch_ratio(df)
        df = self._add_label(df)
        df = self._dedup(df)
        df = self._process_timestamp(df)
        df = self._filter_cold_start(df)
        df = self._sort_by_time(df)

        logger.info(
            f"清洗完成: {len(df):,} 行  "
            f"用户: {df['user_id'].nunique():,}  "
            f"视频: {df['item_id'].nunique():,}  "
            f"正样本率: {df['label'].mean():.3f}"
        )
        return df

    # ─────────────────────────────────────────
    # 私有方法
    # ─────────────────────────────────────────

    def _unify_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """统一列名：video_id → item_id"""
        if 'video_id' in df.columns:
            df = df.rename(columns={'video_id': 'item_id'})
            logger.info("列名统一: video_id → item_id")
        return df

    def _process_watch_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        watch_ratio 处理
        - 填充缺失值为 0
        - clip 到 [0, clip_val]（watch_ratio > 1 表示重复观看，保留但限制上限）
        """
        df = df.copy()
        df['watch_ratio'] = df['watch_ratio'].fillna(0).clip(0, self.clip_val)
        logger.info(
            f"watch_ratio 处理完成  "
            f"mean={df['watch_ratio'].mean():.3f}  "
            f"clip_val={self.clip_val}"
        )
        return df

    def _add_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加二值标签：watch_ratio >= threshold → 正样本"""
        df = df.copy()
        df['label'] = (df['watch_ratio'] >= self.threshold).astype(int)
        pos_rate = df['label'].mean()
        logger.info(
            f"正样本标签添加完成  "
            f"阈值={self.threshold}  正样本率={pos_rate:.3f}"
        )
        return df

    def _dedup(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        去重：同一用户对同一视频保留 watch_ratio 最大的一条
        KuaiRec 场景：同一视频可能被多次曝光
        """
        before = len(df)
        df = df.sort_values('watch_ratio', ascending=False)
        df = df.drop_duplicates(subset=['user_id', 'item_id'], keep='first')
        logger.info(f"去重: {before:,} → {len(df):,}  (-{before - len(df):,})")
        return df

    def _process_timestamp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        时间戳处理
        - 有时间戳列（time/timestamp）→ 转为 Unix 时间戳
        - 无时间戳（small_matrix）→ 用行索引生成伪时序
        """
        df = df.copy()
        ts_col = None
        for col in ['time', 'timestamp', 'date']:
            if col in df.columns:
                ts_col = col
                break

        if ts_col is not None:
            df['timestamp_unix'] = pd.to_numeric(df[ts_col], errors='coerce').fillna(0)
            logger.info(f"时间戳处理: 使用列 '{ts_col}'")
        else:
            # small_matrix 无时间戳，用行索引生成伪时序
            df = df.reset_index(drop=True)
            df['timestamp_unix'] = df.index
            logger.info("无时间戳列，使用行索引生成伪时序")

        return df

    def _filter_cold_start(self, df: pd.DataFrame) -> pd.DataFrame:
        """冷启动过滤"""
        before = len(df)

        user_cnt = df['user_id'].value_counts()
        item_cnt = df['item_id'].value_counts()

        df = df[
            df['user_id'].isin(user_cnt[user_cnt >= self.min_user].index) &
            df['item_id'].isin(item_cnt[item_cnt >= self.min_item].index)
        ]
        logger.info(
            f"冷启动过滤: {before:,} → {len(df):,}  "
            f"(-{before - len(df):,})"
        )
        return df

    def _sort_by_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """按用户+时间排序（保证时序一致性）"""
        df = df.sort_values(['user_id', 'timestamp_unix']).reset_index(drop=True)
        logger.info("时序排序完成")
        return df
