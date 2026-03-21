# run_step1.py
# Step1 工程化入口脚本
# 一行命令跑完完整 Step1 Pipeline
#
# 用法:
#   python run_step1.py
#   python run_step1.py --base_dir /content/VideoRecSys
#   python run_step1.py --threshold 1.0  # 使用更严格的正样本阈值
# ============================================================

import os
import argparse
import logging
import pandas as pd
import numpy as np

from pipeline.data_cleaning       import DataCleaningPipeline
from pipeline.feature_engineering import (UserFeatureEngineer,
                                           ItemFeatureEngineer,
                                           SequenceFeatureEngineer)
from pipeline.sample_factory      import SampleFactory
from pipeline.feature_store       import FeatureStore
from pipeline.dqc_monitor         import (DataQualityChecker,
                                           PSIMonitor,
                                           SampleDistributionChecker)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('Step1')


def parse_args():
    parser = argparse.ArgumentParser(description='Step1 KuaiRec Pipeline')
    parser.add_argument('--base_dir',  default='/content/VideoRecSys')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='watch_ratio 正样本阈值（默认0.5）')
    parser.add_argument('--small',     action='store_true', default=True,
                        help='使用 small_matrix（默认）')
    parser.add_argument('--neg_ratio', type=int, default=4)
    parser.add_argument('--seed',      type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 路径 ──
    raw_dir   = os.path.join(args.base_dir, 'data', 'raw', 'kuairec')
    paths = {
        'cleaned'      : os.path.join(args.base_dir, 'data', 'cleaned'),
        'features'     : os.path.join(args.base_dir, 'data', 'features'),
        'samples'      : os.path.join(args.base_dir, 'data', 'samples'),
        'feature_store': os.path.join(args.base_dir, 'feature_store'),
        'reports'      : os.path.join(args.base_dir, 'reports'),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    # ── 加载原始数据 ──
    matrix = 'small_matrix.csv' if args.small else 'big_matrix.csv'
    log.info(f"加载原始数据: {matrix}")
    inter_df   = pd.read_csv(os.path.join(raw_dir, matrix))
    user_meta  = pd.read_csv(os.path.join(raw_dir, 'user_features.csv'))
    item_daily = pd.read_csv(os.path.join(raw_dir, 'item_daily_features.csv'))
    item_cat   = pd.read_csv(os.path.join(raw_dir, 'item_categories.csv'))
    log.info(f"原始交互数: {len(inter_df):,}")

    # ── Step1: 数据清洗 ──
    log.info("=" * 50)
    log.info("Step1: 数据清洗")
    cleaner = DataCleaningPipeline(
        watch_ratio_threshold=args.threshold,
        min_user_interactions=10,
        min_item_interactions=5,
    )
    df = cleaner.run(inter_df)
    df.to_parquet(
        os.path.join(paths['cleaned'], 'interactions_cleaned.parquet'),
        index=False, compression='snappy')

    # ── Step2: 数据质量检查 ──
    log.info("=" * 50)
    log.info("Step2: 数据质量检查")
    dqc = DataQualityChecker()
    dqc.run(df)

    # ── Step3: 特征工程 ──
    log.info("=" * 50)
    log.info("Step3: 特征工程")

    user_eng  = UserFeatureEngineer()
    user_feat = user_eng.run(df, users_meta=user_meta)
    user_feat.to_parquet(
        os.path.join(paths['features'], 'user_features.parquet'),
        index=False, compression='snappy')

    item_eng  = ItemFeatureEngineer()
    item_feat = item_eng.run(df, item_categories=item_cat, item_daily=item_daily)
    item_feat.to_parquet(
        os.path.join(paths['features'], 'item_features.parquet'),
        index=False, compression='snappy')

    seq_eng = SequenceFeatureEngineer(seq_len=50)
    seq_df  = seq_eng.run(df)
    seq_df.to_parquet(
        os.path.join(paths['features'], 'user_sequences.parquet'),
        index=False, compression='snappy')

    # ── Step4: 样本生成 ──
    log.info("=" * 50)
    log.info("Step4: 时序划分 + 负采样")

    factory   = SampleFactory(
        test_ratio=0.2, neg_ratio=args.neg_ratio, random_seed=args.seed)
    train_pos, test_df_raw = factory.temporal_split(df)

    all_items      = list(df['item_id'].unique())
    item_popularity = df['item_id'].value_counts()
    train_df = factory.negative_sampling(train_pos, all_items, item_popularity)
    test_df  = test_df_raw.copy()

    train_df.to_parquet(
        os.path.join(paths['samples'], 'train.parquet'),
        index=False, compression='snappy')
    test_df.to_parquet(
        os.path.join(paths['samples'], 'test.parquet'),
        index=False, compression='snappy')

    # 样本分布检查
    dist_checker = SampleDistributionChecker()
    dist_checker.check(train_df, test_df)

    # ── Step5: Feature Store 写入 ──
    log.info("=" * 50)
    log.info("Step5: Feature Store 写入")

    fs = FeatureStore(paths['feature_store'])
    fs.write_users(user_feat)
    fs.write_items(item_feat)
    fs.write_sequences(seq_df)
    fs.save_snapshot()

    # ── Step6: PSI 监控 ──
    log.info("=" * 50)
    log.info("Step6: PSI 特征漂移监控")

    psi_monitor = PSIMonitor()
    train_uf = user_feat[user_feat['user_id'].isin(train_df['user_id'].unique())]
    test_uf  = user_feat[user_feat['user_id'].isin(test_df['user_id'].unique())]
    monitor_feats = ['interaction_count', 'avg_watch_ratio',
                     'positive_rate', 'activity_tier', 'std_watch_ratio']
    monitor_feats = [f for f in monitor_feats if f in user_feat.columns]
    psi_report = psi_monitor.monitor(train_uf, test_uf, monitor_feats)
    psi_monitor.monitor_watch_ratio(train_df, test_df)

    # ── 完成报告 ──
    log.info("=" * 50)
    log.info("✅  Step1 Pipeline 完成！")
    log.info(f"  清洗后交互: {len(df):,}")
    log.info(f"  用户数:     {df['user_id'].nunique():,}")
    log.info(f"  视频数:     {df['item_id'].nunique():,}")
    log.info(f"  训练样本:   {len(train_df):,}")
    log.info(f"  测试样本:   {len(test_df):,}")
    log.info(f"  用户特征:   {user_feat.shape[1]-1} 列")
    log.info(f"  视频特征:   {item_feat.shape[1]-1} 列")
    fs_stats = fs.stats()
    log.info(f"  Feature Store: {fs_stats['total_keys']:,} keys")


if __name__ == '__main__':
    main()
