# run_step3.py
# Step3 工程化入口脚本
# 一行命令跑完完整 Step3 DIN 精排 Pipeline
#
# 用法:
#   python run_step3.py
#   python run_step3.py --base_dir /content/VideoRecSys --epochs 30
# ============================================================

import os
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import faiss
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from models.ranking import (DINMultiTask, RankingDataset,
                             RankingTrainer, RankingEvaluator,
                             RankingPredictor)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('Step3')


def parse_args():
    p = argparse.ArgumentParser(description='Step3 DIN 精排 Pipeline')
    p.add_argument('--base_dir',       default='/content/VideoRecSys')
    p.add_argument('--embedding_dim',  type=int,   default=32)
    p.add_argument('--batch_size',     type=int,   default=2048)
    p.add_argument('--epochs',         type=int,   default=30)
    p.add_argument('--lr',             type=float, default=5e-4)
    p.add_argument('--early_stop',     type=int,   default=4)
    p.add_argument('--alpha',          type=float, default=0.5,
                   help='多任务权重: α*CTR + (1-α)*完播率')
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    paths = {
        'features'     : os.path.join(args.base_dir, 'data', 'features'),
        'samples'      : os.path.join(args.base_dir, 'data', 'samples'),
        'feature_store': os.path.join(args.base_dir, 'feature_store'),
        'recall_models': os.path.join(args.base_dir, 'models', 'recall'),
        'rank_models'  : os.path.join(args.base_dir, 'models', 'ranking'),
        'reports'      : os.path.join(args.base_dir, 'reports'),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    cfg = {
        'embedding_dim'   : args.embedding_dim,
        'attention_hidden': [64, 16],
        'dnn_hidden'      : [256, 128, 64],
        'dropout'         : 0.3,
        'lr'              : args.lr,
        'batch_size'      : args.batch_size,
        'epochs'          : args.epochs,
        'early_stop'      : args.early_stop,
        'seq_len'         : 50,
        'tower_emb_dim'   : 64,
        'watch_ratio_max' : 3.0,
        'multitask_alpha' : args.alpha,
        'device'          : device,
    }

    # ── 加载数据 ──
    log.info("加载 Step1/2 产出...")
    train_df    = pd.read_parquet(os.path.join(paths['samples'],   'train.parquet'))
    test_df     = pd.read_parquet(os.path.join(paths['samples'],   'test.parquet'))
    user_feat   = pd.read_parquet(os.path.join(paths['features'],  'user_features.parquet'))
    item_feat   = pd.read_parquet(os.path.join(paths['features'],  'item_features.parquet'))
    seq_df      = pd.read_parquet(os.path.join(paths['features'],  'user_sequences.parquet'))
    user_emb_df = pd.read_parquet(os.path.join(paths['feature_store'], 'user_embeddings.parquet'))
    item_emb_df = pd.read_parquet(os.path.join(paths['feature_store'], 'item_embeddings.parquet'))
    with open(os.path.join(paths['recall_models'], 'id_mappings.pkl'), 'rb') as f:
        id_maps  = pickle.load(f)
    user2idx = id_maps['user2idx']
    item2idx = id_maps['item2idx']
    idx2item = id_maps['idx2item']

    # ── 特征归一化 ──
    USER_COLS = [c for c in user_feat.columns
                 if c != 'user_id' and user_feat[c].dtype in
                 [np.float64, np.float32, np.int64, np.int32]]
    ITEM_COLS = [c for c in item_feat.columns
                 if c != 'item_id' and item_feat[c].dtype in
                 [np.float64, np.float32, np.int64, np.int32]]

    u_scaler = MinMaxScaler()
    i_scaler = MinMaxScaler()
    u_np = u_scaler.fit_transform(
        user_feat[USER_COLS].fillna(0).values.astype(np.float32))
    i_np = i_scaler.fit_transform(
        item_feat[ITEM_COLS].fillna(0).values.astype(np.float32))

    user_feat_dict = {uid: u_np[i] for i, uid in enumerate(user_feat['user_id'])}
    item_feat_dict = {iid: i_np[i] for i, iid in enumerate(item_feat['item_id'])}

    EMB_COLS_U    = [c for c in user_emb_df.columns if c.startswith('user_emb_')]
    EMB_COLS_I    = [c for c in item_emb_df.columns if c.startswith('item_emb_')]
    user_emb_dict = {row['user_id']: row[EMB_COLS_U].values.astype(np.float32)
                     for _, row in user_emb_df.iterrows()}
    item_emb_dict = {row['item_id']: row[EMB_COLS_I].values.astype(np.float32)
                     for _, row in item_emb_df.iterrows()}

    seq_dict = {}
    for _, row in seq_df.iterrows():
        raw = str(row.get('pos_seq', '') or '')
        ids = [item2idx[int(x)] for x in raw.split(',')
               if x and int(x) in item2idx]
        seq_dict[row['user_id']] = ids

    U_DIM   = len(USER_COLS)
    I_DIM   = len(ITEM_COLS)
    EMB_DIM = len(EMB_COLS_U)
    N_ITEMS = len(item2idx)

    # ── Dataset & DataLoader ──
    train_ds = RankingDataset(
        train_df, user_feat_dict, item_feat_dict,
        user_emb_dict, item_emb_dict,
        seq_dict, item2idx, cfg['seq_len'], cfg['watch_ratio_max'])
    test_ds  = RankingDataset(
        test_df, user_feat_dict, item_feat_dict,
        user_emb_dict, item_emb_dict,
        seq_dict, item2idx, cfg['seq_len'], cfg['watch_ratio_max'])

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(
        test_ds, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True)

    log.info(f"训练集: {len(train_ds):,}  测试集: {len(test_ds):,}")

    # ── 模型 & 训练 ──
    model   = DINMultiTask(N_ITEMS, I_DIM, U_DIM, EMB_DIM, cfg).to(device)
    trainer = RankingTrainer(model, cfg, paths['rank_models'])
    trainer.train(train_loader, test_loader)

    # ── 离线评估 ──
    ctr_preds, dur_preds, ctr_labels, dur_labels = trainer.predict(test_loader)
    evaluator = RankingEvaluator()
    results   = evaluator.evaluate(
        test_df.reset_index(drop=True),
        ctr_preds, dur_preds, ctr_labels, dur_labels)
    evaluator.print_report(results)

    # ── 模型保存（附加 scaler）──
    with open(os.path.join(paths['rank_models'], 'scalers.pkl'), 'wb') as f:
        pickle.dump({'user': u_scaler, 'item': i_scaler}, f)

    # ── 链路测试 ──
    log.info("完整推荐链路测试...")
    fi_index     = faiss.read_index(
        os.path.join(paths['recall_models'], 'faiss_item.index'))
    recall_item_ids = np.load(
        os.path.join(paths['recall_models'], 'item_ids.npy')).tolist()
    recall_user_ids = np.load(
        os.path.join(paths['recall_models'], 'user_ids.npy')).tolist()
    user_id2row  = {uid: i for i, uid in enumerate(recall_user_ids)}

    user_emb_matrix = np.stack([
        user_emb_dict.get(uid, np.zeros(EMB_DIM, np.float32))
        for uid in recall_user_ids
    ]).astype(np.float32)

    predictor = RankingPredictor(
        model, fi_index, user_emb_matrix, user_id2row,
        recall_item_ids, user_feat_dict, item_feat_dict,
        user_emb_dict, item_emb_dict, seq_dict, cfg)

    test_users = list(user2idx.keys())[:3]
    for uid in test_users:
        results_rec = predictor.recommend(uid, recall_k=200, top_n=5)
        log.info(f"用户 {uid} Top5 推荐:")
        for rank, (iid, ctr, dur, score) in enumerate(results_rec, 1):
            log.info(f"  #{rank} item={iid} CTR={ctr:.3f} 完播率={dur:.3f} 综合={score:.3f}")

    # ── 完成报告 ──
    log.info("=" * 50)
    log.info("✅  Step3 Pipeline 完成！")
    log.info(f"  CTR AUC:        {results.get('AUC', 0):.4f}")
    log.info(f"  GAUC:           {results.get('GAUC', 0):.4f}")
    log.info(f"  NDCG@10:        {results.get('NDCG@10', 0):.4f}")
    log.info(f"  完播率 RMSE:    {results.get('WatchRatio_RMSE', 0):.4f}")


if __name__ == '__main__':
    main()
