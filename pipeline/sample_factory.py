# pipeline/sample_factory.py
# 样本工厂模块 — KuaiRec 版本
# 对应 Step1 Cell 8
# ============================================================

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

logger = logging.getLogger(__name__)


class SampleFactory:
    """
    样本工厂 — KuaiRec 版本

    KuaiRec 特有处理:
        - 负样本默认 watch_ratio = 0（未观看）
        - 支持按 watch_ratio 权重采样热门负样本
        - 时序划分严格按每用户的时间顺序

    处理流程:
        1. 时序划分（每用户最后 test_ratio 进测试集）
        2. 负采样（50% 随机 + 50% 热门加权）
        3. 特征拼接
    """

    def __init__(
        self,
        test_ratio: float = 0.2,
        neg_ratio: int = 4,
        random_neg_ratio: float = 0.5,
        hot_pool_size: int = 2000,
        random_seed: int = 42,
    ):
        self.test_ratio       = test_ratio
        self.neg_ratio        = neg_ratio
        self.random_neg_ratio = random_neg_ratio
        self.hot_pool_size    = hot_pool_size
        self.random_seed      = random_seed
        np.random.seed(random_seed)

    def temporal_split(
        self,
        interactions: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        时序划分
        每个用户按 timestamp_unix 排序后，最后 test_ratio 进测试集

        Returns:
            (train_df, test_df) 只包含正样本，负采样在后续步骤进行
        """
        logger.info(f"时序划分  test_ratio={self.test_ratio}")
        train_list, test_list = [], []

        for uid, group in interactions.groupby('user_id'):
            group  = group.sort_values('timestamp_unix')
            n_test = max(1, int(len(group) * self.test_ratio))
            train_list.append(group.iloc[:-n_test])
            test_list.append(group.iloc[-n_test:])

        train_df = pd.concat(train_list).reset_index(drop=True)
        test_df  = pd.concat(test_list).reset_index(drop=True)

        logger.info(
            f"  训练集: {len(train_df):,}  "
            f"正样本率: {train_df['label'].mean():.3f}"
        )
        logger.info(
            f"  测试集: {len(test_df):,}  "
            f"正样本率: {test_df['label'].mean():.3f}"
        )
        return train_df, test_df

    def negative_sampling(
        self,
        pos_df: pd.DataFrame,
        all_item_ids: List,
        item_popularity: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        负采样（只对训练集做）
        策略: 50% 随机 + 50% 热门加权

        Args:
            pos_df:          正样本 DataFrame
            all_item_ids:    全量视频 ID 列表
            item_popularity: 视频热度（value_counts），用于热门采样

        Returns:
            包含正负样本的 DataFrame
        """
        logger.info(f"负采样  正:负=1:{self.neg_ratio}")
        all_item_set = set(all_item_ids)

        # 热门候选池
        if item_popularity is not None:
            hot_pool = item_popularity.nlargest(self.hot_pool_size).index.tolist()
        else:
            hot_pool = all_item_ids[:self.hot_pool_size]

        neg_records = []
        for uid, group in pos_df.groupby('user_id'):
            interacted = set(group['item_id'].tolist())
            n_neg      = len(group) * self.neg_ratio

            # 随机负样本
            cands   = list(all_item_set - interacted)
            n_rand  = min(int(n_neg * self.random_neg_ratio), len(cands))
            rand_negs = np.random.choice(cands, size=n_rand, replace=False)

            # 热门负样本
            hot_cands = [i for i in hot_pool if i not in interacted]
            n_hot   = min(int(n_neg * (1 - self.random_neg_ratio)), len(hot_cands))
            hot_negs  = np.random.choice(hot_cands, size=n_hot, replace=False)

            ts_max = group['timestamp_unix'].max()
            for iid in list(rand_negs) + list(hot_negs):
                neg_records.append({
                    'user_id'       : uid,
                    'item_id'       : iid,
                    'watch_ratio'   : 0.0,
                    'label'         : 0,
                    'timestamp_unix': ts_max,
                })

        neg_df = pd.DataFrame(neg_records)
        result = pd.concat([pos_df, neg_df]).sample(
            frac=1, random_state=self.random_seed
        ).reset_index(drop=True)

        logger.info(
            f"  总样本: {len(result):,}  "
            f"正样本率: {result['label'].mean():.3f}"
        )
        return result

    def join_features(
        self,
        samples: pd.DataFrame,
        user_feat: pd.DataFrame,
        item_feat: pd.DataFrame,
    ) -> pd.DataFrame:
        """拼接用户和视频特征"""
        logger.info("拼接特征...")
        result = samples.merge(user_feat, on='user_id', how='left')
        result = result.merge(
            item_feat, on='item_id', how='left', suffixes=('_u', '_i'))
        logger.info(f"  样本维度: {result.shape}")
        return result
