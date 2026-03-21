# run_step4.py
# Step4 工程化入口脚本
# A/B 实验平台 + 在线监控 + 项目总结报告
#
# 用法:
#   python run_step4.py
#   python run_step4.py --base_dir /content/VideoRecSys
# ============================================================

import os
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import faiss
from tabulate import tabulate

from experiment  import ABFramework, MetricsCalculator, StatisticalTester
from monitoring  import PSIMonitor, ServingMonitor

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('Step4')


def parse_args():
    p = argparse.ArgumentParser(description='Step4 A/B 实验 + 监控')
    p.add_argument('--base_dir', default='/content/VideoRecSys')
    p.add_argument('--n_bench',  type=int, default=200,
                   help='压测用户数')
    return p.parse_args()


def main():
    args = parse_args()

    paths = {
        'features'     : os.path.join(args.base_dir, 'data', 'features'),
        'samples'      : os.path.join(args.base_dir, 'data', 'samples'),
        'feature_store': os.path.join(args.base_dir, 'feature_store'),
        'recall_models': os.path.join(args.base_dir, 'models', 'recall'),
        'rank_models'  : os.path.join(args.base_dir, 'models', 'ranking'),
        'reports'      : os.path.join(args.base_dir, 'reports'),
    }

    # ── 加载数据 ──
    log.info("加载前序产出...")
    train_df    = pd.read_parquet(os.path.join(paths['samples'],   'train.parquet'))
    test_df     = pd.read_parquet(os.path.join(paths['samples'],   'test.parquet'))
    user_feat   = pd.read_parquet(os.path.join(paths['features'],  'user_features.parquet'))
    item_feat   = pd.read_parquet(os.path.join(paths['features'],  'item_features.parquet'))
    user_emb_df = pd.read_parquet(os.path.join(paths['feature_store'], 'user_embeddings.parquet'))

    with open(os.path.join(paths['recall_models'], 'id_mappings.pkl'), 'rb') as f:
        id_maps  = pickle.load(f)
    user2idx = id_maps['user2idx']

    # ═══════════════════════════════════════
    # Step1: A/B 实验分流
    # ═══════════════════════════════════════
    log.info("=" * 55)
    log.info("Step1: A/B 实验分流")

    ab        = ABFramework()
    all_users = list(test_df['user_id'].unique())

    # KuaiRec 3 个并行实验
    ab.create_experiment('exp_recall',    0.5, '热门召回 vs 双塔召回')
    ab.create_experiment('exp_ranking',   0.5, '评分排序 vs DIN多任务')
    ab.create_experiment('exp_objective', 0.5, '纯CTR vs CTR+完播率融合')

    ctrl_recall,    trt_recall    = ab.split_users(all_users, 'exp_recall')
    ctrl_ranking,   trt_ranking   = ab.split_users(all_users, 'exp_ranking')
    ctrl_objective, trt_objective = ab.split_users(all_users, 'exp_objective')

    # 打印分桶结果
    print('\n【A/B 实验分桶结果】')
    rows = []
    for exp_id in ab.experiments:
        s = ab.get_stats(exp_id)
        rows.append([
            exp_id,
            f"{s['control_n']:,} ({s['control_pct']:.1f}%)",
            f"{s['treatment_n']:,} ({100-s['control_pct']:.1f}%)",
            ab.experiments[exp_id].desc,
        ])
    print(tabulate(rows, headers=['实验ID', '对照组', '实验组', '说明'], tablefmt='grid'))

    # 正交性验证
    ab.verify_orthogonality(
        all_users[:200],
        ['exp_recall', 'exp_ranking', 'exp_objective'])

    # ═══════════════════════════════════════
    # Step2: 实验效果评估 + 统计检验
    # ═══════════════════════════════════════
    log.info("=" * 55)
    log.info("Step2: 实验效果评估 + 统计显著性检验")

    calculator = MetricsCalculator()
    tester     = StatisticalTester()
    metrics    = ['ctr', 'avg_watch', 'ndcg']

    # 计算各组指标
    ctrl_r_m = calculator.compute_user_metrics(test_df, ctrl_recall)
    trt_r_m  = calculator.compute_user_metrics(test_df, trt_recall)
    ctrl_k_m = calculator.compute_user_metrics(test_df, ctrl_ranking)
    trt_k_m  = calculator.compute_user_metrics(test_df, trt_ranking)
    ctrl_o_m = calculator.compute_user_metrics(test_df, ctrl_objective)
    trt_o_m  = calculator.compute_user_metrics(test_df, trt_objective)

    # 统计检验
    tester.run_experiment_analysis(
        ctrl_r_m, trt_r_m, metrics,
        '召回策略实验', '热门召回', '双塔召回')
    tester.run_experiment_analysis(
        ctrl_k_m, trt_k_m, metrics,
        '精排策略实验', '评分排序', 'DIN多任务')
    tester.run_experiment_analysis(
        ctrl_o_m, trt_o_m, metrics,
        '排序目标实验', '纯CTR',    'CTR+完播率')

    # ═══════════════════════════════════════
    # Step3: PSI 特征漂移监控
    # ═══════════════════════════════════════
    log.info("=" * 55)
    log.info("Step3: PSI 特征漂移监控")

    psi_monitor = PSIMonitor()
    train_uf = user_feat[user_feat['user_id'].isin(train_df['user_id'].unique())]
    test_uf  = user_feat[user_feat['user_id'].isin(test_df['user_id'].unique())]

    monitor_feats = ['interaction_count', 'avg_watch_ratio',
                     'positive_rate', 'activity_tier', 'std_watch_ratio']
    monitor_feats = [f for f in monitor_feats if f in user_feat.columns]

    psi_report = psi_monitor.monitor_features(train_uf, test_uf, monitor_feats)
    psi_monitor.monitor_watch_ratio(train_df, test_df)

    print('\n【PSI 特征监控报告】')
    print(tabulate(
        psi_report[['feature', 'psi', 'status', 'action']].values.tolist(),
        headers=['特征', 'PSI', '状态', '建议'],
        tablefmt='grid'))

    # ═══════════════════════════════════════
    # Step4: 链路压测
    # ═══════════════════════════════════════
    log.info("=" * 55)
    log.info("Step4: 推荐链路压测")

    svc_monitor = ServingMonitor()

    try:
        fi_index     = faiss.read_index(
            os.path.join(paths['recall_models'], 'faiss_item.index'))
        recall_user_ids = np.load(
            os.path.join(paths['recall_models'], 'user_ids.npy')).tolist()
        user_id2row  = {uid: i for i, uid in enumerate(recall_user_ids)}

        EMB_COLS_U   = [c for c in user_emb_df.columns if c.startswith('user_emb_')]
        user_emb_dict = {row['user_id']: row[EMB_COLS_U].values.astype(np.float32)
                         for _, row in user_emb_df.iterrows()}
        EMB_DIM = len(EMB_COLS_U)

        user_emb_matrix = np.stack([
            user_emb_dict.get(uid, np.zeros(EMB_DIM, np.float32))
            for uid in recall_user_ids
        ]).astype(np.float32)
        faiss.normalize_L2(user_emb_matrix)

        bench_users = [u for u in list(user2idx.keys())[:args.n_bench]
                       if u in user_id2row]
        recall_stats = svc_monitor.benchmark_recall(
            fi_index, user_emb_matrix, user_id2row, bench_users)
        svc_monitor.print_report(recall_stats, '双塔召回')
    except Exception as e:
        log.warning(f"压测跳过: {e}")
        recall_stats = {'p50': 0, 'p99': 0}

    # ═══════════════════════════════════════
    # Step5: 项目总结报告
    # ═══════════════════════════════════════
    log.info("=" * 55)
    _print_final_report(
        user_feat, item_feat, train_df, test_df,
        psi_report, recall_stats)


def _print_final_report(user_feat, item_feat, train_df, test_df,
                         psi_report, recall_stats):
    print('\n' + '=' * 68)
    print('  🎬  VideoRecSys 项目总结报告（KuaiRec 版本）')
    print('=' * 68)

    print('\n【数据规模】')
    print(tabulate([
        ['数据集',    'KuaiRec small_matrix（快手真实短视频）'],
        ['用户数',    f"{user_feat.shape[0]:,}"],
        ['视频数',    f"{item_feat.shape[0]:,}"],
        ['训练样本',  f"{len(train_df):,}"],
        ['测试样本',  f"{len(test_df):,}"],
        ['正样本定义', 'watch_ratio >= 0.5（看了50%以上）'],
    ], tablefmt='simple'))

    print('\n【4步架构】')
    print(tabulate([
        ['Step1', '数据Pipeline',   'KuaiRec清洗 + watch_ratio特征 + 负采样'],
        ['Step2', '双塔召回',       'WeightedInfoNCE + FAISS + watch_ratio加权'],
        ['Step3', 'DIN精排',        '多任务：CTR预估 + 完播率预估'],
        ['Step4', '实验平台+监控',  'A/B分流 + T-test + PSI漂移监控'],
    ], headers=['步骤', '模块', '核心技术'], tablefmt='simple'))

    print('\n【技术栈对应关系】')
    print(tabulate([
        ['数据处理',  'Pandas/PyArrow',    '企业: Spark/Flink'],
        ['特征存储',  'Parquet/Dict',      '企业: Redis/Hive'],
        ['召回',      'FAISS IndexFlatIP', '企业: Milvus/Proxima'],
        ['精排',      'PyTorch DIN多任务', '企业: TF Serving多目标'],
        ['实验平台',  'Hash分桶+T-test',   '企业: 阿里/字节实验平台'],
        ['监控',      'PSI+分布监控',      '企业: Grafana/Prometheus'],
        ['数据集',    'KuaiRec（快手）',   '企业: 真实线上日志'],
    ], headers=['模块', '本项目', '企业对应'], tablefmt='simple'))

    print('\n【完整推荐链路】')
    print('  用户请求')
    print('    → 双塔召回（FAISS ANN，200候选，watch_ratio加权训练）')
    print('    → DIN多任务精排（CTR + 完播率，注意力机制）')
    print('    → 融合排序（0.5×CTR + 0.5×完播率）')
    print('    → Top 10 推荐结果')

    print('\n' + '=' * 68)
    print('  🎉  全部 4 个 Step 工程化完成！')
    print('  📁  GitHub: github.com/ganm30261-hub/VideoRecSys')
    print('=' * 68)


if __name__ == '__main__':
    main()
