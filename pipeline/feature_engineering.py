# pipeline/feature_engineering.py
# 特征工程模块 — KuaiRec 版本
# 对应 Step1 Cell 5 / Cell 6 / Cell 7
# ============================================================

import logging
import pandas as pd
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


class UserFeatureEngineer:
    """
    用户侧特征工程 — KuaiRec 版本

    生成特征:
        行为统计: interaction_count, avg_watch_ratio, std_watch_ratio,
                  positive_rate, max_watch_ratio
        活跃分层: activity_tier (0-4)
        侧信息:   来自 user_features.csv 的设备/地理特征（数十列）
    """

    ACTIVITY_BINS = [0, 10, 50, 200, 500, float('inf')]
    ACTIVITY_LABELS = [0, 1, 2, 3, 4]

    def run(
        self,
        interactions: pd.DataFrame,
        users_meta: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Args:
            interactions: 清洗后的交互数据
            users_meta:   KuaiRec user_features.csv（可选）

        Returns:
            用户特征 DataFrame
        """
        logger.info("开始用户特征工程（KuaiRec）...")
        feat = self._behavioral_features(interactions)

        if users_meta is not None:
            feat = self._merge_kuairec_features(feat, users_meta)

        # 填充缺失值
        feat = feat.fillna(0)

        logger.info(
            f"用户特征完成: {feat.shape[0]:,} 用户  {feat.shape[1]} 列"
        )
        return feat

    def _behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算用户行为统计特征"""
        agg = df.groupby('user_id').agg(
            interaction_count =('item_id',       'count'),
            avg_watch_ratio   =('watch_ratio',   'mean'),
            std_watch_ratio   =('watch_ratio',   'std'),
            positive_rate     =('label',         'mean'),
            max_watch_ratio   =('watch_ratio',   'max'),
        ).reset_index()

        agg['std_watch_ratio'] = agg['std_watch_ratio'].fillna(0)

        # 活跃度分层
        agg['activity_tier'] = pd.cut(
            agg['interaction_count'],
            bins=self.ACTIVITY_BINS,
            labels=self.ACTIVITY_LABELS,
        ).astype(int)

        return agg

    def _merge_kuairec_features(
        self,
        feat: pd.DataFrame,
        users_meta: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        合并 KuaiRec user_features.csv
        只保留数值型特征，丢弃全空列
        """
        uf = users_meta.copy()

        # 统一 ID 列名
        if 'user_id' not in uf.columns:
            uf = uf.rename(columns={uf.columns[0]: 'user_id'})

        # 只保留数值型列
        num_cols = [
            c for c in uf.columns
            if c != 'user_id' and
            uf[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        # 丢弃全空列
        num_cols = [c for c in num_cols if uf[c].notna().sum() > 0]
        uf = uf[['user_id'] + num_cols].fillna(0)

        feat = feat.merge(uf, on='user_id', how='left')
        logger.info(f"合并 KuaiRec 用户特征: {len(num_cols)} 列")
        return feat


class ItemFeatureEngineer:
    """
    视频侧特征工程 — KuaiRec 版本

    生成特征:
        统计特征:  play_count, avg_watch_ratio, ctr_proxy,
                   std_watch_ratio, popularity_tier
        类别特征:  来自 item_categories.csv 的多热编码
        每日特征:  来自 item_daily_features.csv 的均值聚合
    """

    def run(
        self,
        interactions: pd.DataFrame,
        item_categories: Optional[pd.DataFrame] = None,
        item_daily: Optional[pd.DataFrame] = None,
        max_cat_cols: int = 20,
    ) -> pd.DataFrame:
        """
        Args:
            interactions:    清洗后的交互数据
            item_categories: KuaiRec item_categories.csv（可选）
            item_daily:      KuaiRec item_daily_features.csv（可选）
            max_cat_cols:    最多保留的类别特征数量

        Returns:
            视频特征 DataFrame
        """
        logger.info("开始视频特征工程（KuaiRec）...")
        feat = self._statistical_features(interactions)

        if item_categories is not None:
            feat = self._merge_categories(feat, item_categories, max_cat_cols)

        if item_daily is not None:
            feat = self._merge_daily_features(feat, item_daily)

        # 只保留有效视频，填充缺失值
        valid_items = interactions['item_id'].unique()
        feat = feat[feat['item_id'].isin(valid_items)].fillna(0)

        logger.info(
            f"视频特征完成: {feat.shape[0]:,} 视频  {feat.shape[1]} 列"
        )
        return feat

    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算视频统计特征"""
        agg = df.groupby('item_id').agg(
            play_count      =('user_id',     'count'),
            avg_watch_ratio =('watch_ratio', 'mean'),
            ctr_proxy       =('label',       'mean'),
            std_watch_ratio =('watch_ratio', 'std'),
        ).reset_index()

        agg['std_watch_ratio'] = agg['std_watch_ratio'].fillna(0)

        # 热度分层（自动适配实际分桶数）
        _, bin_edges = pd.qcut(
            agg['play_count'], q=5, retbins=True, duplicates='drop')
        n_bins = len(bin_edges) - 1
        agg['popularity_tier'] = pd.qcut(
            agg['play_count'],
            q=5,
            labels=list(range(n_bins)),
            duplicates='drop',
        ).astype(int)

        return agg

    def _merge_categories(
        self,
        feat: pd.DataFrame,
        item_cat: pd.DataFrame,
        max_cols: int,
    ) -> pd.DataFrame:
        """合并视频类别特征（多热编码）"""
        ic = item_cat.copy()
        id_col = 'video_id' if 'video_id' in ic.columns else ic.columns[0]
        ic = ic.rename(columns={id_col: 'item_id'})

        # 找到类别列
        cat_cols = [
            c for c in ic.columns
            if 'tag' in c.lower() or 'cat' in c.lower() or 'feat' in c.lower()
        ]
        if not cat_cols:
            logger.warning("未找到类别列，跳过类别特征")
            return feat

        cat_col = cat_cols[0]
        try:
            dummies = ic.set_index('item_id')[cat_col].str.get_dummies(sep=',')
            dummies.columns = [f'cat_{c}' for c in dummies.columns[:max_cols]]
            dummies = dummies.iloc[:, :max_cols].reset_index()
            feat = feat.merge(dummies, on='item_id', how='left')
            logger.info(f"合并类别特征: {min(len(dummies.columns)-1, max_cols)} 列")
        except Exception as e:
            logger.warning(f"类别特征合并失败: {e}")

        return feat

    def _merge_daily_features(
        self,
        feat: pd.DataFrame,
        item_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """合并视频每日统计特征（按视频取均值）"""
        id_col = 'video_id' if 'video_id' in item_daily.columns else item_daily.columns[0]
        df = item_daily.rename(columns={id_col: 'item_id'})

        num_cols = [
            c for c in df.columns
            if c != 'item_id' and
            df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        agg = df.groupby('item_id')[num_cols].mean().reset_index()
        feat = feat.merge(agg, on='item_id', how='left')
        logger.info(f"合并每日特征: {len(num_cols)} 列")
        return feat


class SequenceFeatureEngineer:
    """
    用户行为序列特征工程 — KuaiRec 版本

    生成:
        all_seq:     最近 seq_len 条交互的 item_id 序列
        pos_seq:     最近 seq_len 条正反馈（watch_ratio >= threshold）的 item_id 序列
        seq_len:     实际序列长度
        pos_seq_len: 正序列实际长度
    """

    def __init__(self, seq_len: int = 50):
        self.seq_len = seq_len

    def run(self, interactions: pd.DataFrame) -> pd.DataFrame:
        """
        Args:
            interactions: 时序排序后的清洗数据

        Returns:
            用户序列特征 DataFrame
        """
        logger.info(f"开始序列特征工程  seq_len={self.seq_len}...")
        records = []

        for uid, group in interactions.groupby('user_id'):
            group = group.sort_values('timestamp_unix')

            all_seq = group['item_id'].tolist()[-self.seq_len:]
            pos_seq = group[group['label'] == 1]['item_id'].tolist()[-self.seq_len:]

            records.append({
                'user_id'    : uid,
                'all_seq'    : ','.join(map(str, all_seq)),
                'pos_seq'    : ','.join(map(str, pos_seq)),
                'seq_len'    : len(all_seq),
                'pos_seq_len': len(pos_seq),
            })

        seq_df = pd.DataFrame(records)
        logger.info(
            f"序列特征完成: {len(seq_df):,} 用户  "
            f"平均序列长度: {seq_df['seq_len'].mean():.1f}  "
            f"平均正序列长度: {seq_df['pos_seq_len'].mean():.1f}"
        )
        return seq_df
