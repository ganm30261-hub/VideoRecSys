# run_step2.py
# Step2 工程化入口脚本
# 一行命令跑完完整 Step2 双塔召回 Pipeline
#
# 用法:
#   python run_step2.py
#   python run_step2.py --base_dir /content/VideoRecSys --epochs 20
# ============================================================

import os
import argparse
import logging
import pickle
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from models.recall import (TwoTowerModel, RecallDataset,
                            FaissIndex, RecallTrainer, RecallEvaluator)

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(name)s %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger('Step2')


def parse_args():
    p = argparse.ArgumentParser(description='Step2 双塔召回 Pipeline')
    p.add_argument('--base_dir',    default='/content/VideoRecSys')
    p.add_argument('--embedding_dim', type=int,   default=64)
    p.add_argument('--batch_size',    type=int,   default=2048)
    p.add_argument('--epochs',        type=int,   default=20)
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--early_stop',    type=int,   default=3)
    p.add_argument('--temperature',   type=float, default=0.07)
    p.add_argument('--seed',          type=int,   default=42)
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
        'reports'      : os.path.join(args.base_dir, 'reports'),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)

    cfg = {
        'embedding_dim': args.embedding_dim,
        'hidden_layers': [256, 128, 64],
        'dropout'      : 0.2,
        'temperature'  : args.temperature,
        'lr'           : args.lr,
        'batch_size'   : args.batch_size,
        'epochs'       : args.epochs,
        'early_stop'   : args.early_stop,
        'seq_len'      : 50,
        'device'       : device,
    }

    # ── 加载数据 ──
    log.info("加载 Step1 产出...")
    train_df  = pd.read_parquet(os.path.join(paths['samples'], 'train.parquet'))
    test_df   = pd.read_parquet(os.path.join(paths['samples'], 'test.parquet'))
    user_feat = pd.read_parquet(os.path.join(paths['features'], 'user_features.parquet'))
    item_feat = pd.read_parquet(os.path.join(paths['features'], 'item_features.parquet'))
    seq_df    = pd.read_parquet(os.path.join(paths['features'], 'user_sequences.parquet'))

    # ── ID 映射 ──
    all_users = sorted(user_feat['user_id'].unique())
    all_items = sorted(item_feat['item_id'].unique())
    user2idx  = {u: i for i, u in enumerate(all_users)}
    item2idx  = {v: i for i, v in enumerate(all_items)}
    idx2user  = {i: u for u, i in user2idx.items()}
    idx2item  = {i: v for v, i in item2idx.items()}

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

    # ── 序列字典 ──
    seq_dict = {}
    for _, row in seq_df.iterrows():
        raw = str(row.get('pos_seq', '') or '')
        ids = [item2idx[int(x)] for x in raw.split(',')
               if x and int(x) in item2idx]
        seq_dict[row['user_id']] = ids

    U_DIM    = len(USER_COLS)
    I_DIM    = len(ITEM_COLS)
    N_USERS  = len(user2idx)
    N_ITEMS  = len(item2idx)

    # ── Dataset & DataLoader ──
    train_ds = RecallDataset(
        train_df, user_feat_dict, item_feat_dict,
        user2idx, item2idx, seq_dict, cfg['seq_len'])
    test_ds  = RecallDataset(
        test_df, user_feat_dict, item_feat_dict,
        user2idx, item2idx, seq_dict, cfg['seq_len'])

    train_loader = DataLoader(
        train_ds, batch_size=cfg['batch_size'],
        shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(
        test_ds, batch_size=cfg['batch_size'],
        shuffle=False, num_workers=2, pin_memory=True)

    log.info(f"训练正样本: {len(train_ds):,}  验证集: {len(test_ds):,}")

    # ── 模型 & 训练 ──
    model   = TwoTowerModel(N_USERS, N_ITEMS, U_DIM, I_DIM, cfg).to(device)
    trainer = RecallTrainer(model, cfg, paths['recall_models'])
    history = trainer.train(train_loader, val_loader)
    trainer.save_id_mappings(user2idx, item2idx, idx2user, idx2item)

    # ── 生成全量 Embedding ──
    user_ids  = list(user2idx.keys())
    item_ids  = list(item2idx.keys())
    user_embs, item_embs = trainer.generate_embeddings(
        user_ids, item_ids,
        user_feat_dict, item_feat_dict,
        seq_dict, user2idx, item2idx)
    log.info(f"Embedding 生成: user={user_embs.shape}  item={item_embs.shape}")

    # ── FAISS 索引 ──
    fi = FaissIndex(cfg['embedding_dim'], N_ITEMS)
    fi.build(item_embs, item_ids)
    fi.save(paths['recall_models'])
    np.save(os.path.join(paths['recall_models'], 'user_ids.npy'),
            np.array(user_ids))

    # ── 评估 ──
    evaluator = RecallEvaluator()
    metrics   = evaluator.evaluate(
        test_df, user_embs, item_embs, user_ids, item_ids, fi)
    baselines = evaluator.compare_baselines(test_df, item_ids)

    log.info("=== 召回评估结果 ===")
    for k, v in metrics.items():
        log.info(f"  {k}: {v:.4f}")

    # ── 写入 Feature Store ──
    emb_dim = user_embs.shape[1]
    import pandas as pd
    user_emb_df = pd.DataFrame(
        user_embs, columns=[f'user_emb_{i}' for i in range(emb_dim)])
    user_emb_df.insert(0, 'user_id', user_ids)
    item_emb_df = pd.DataFrame(
        item_embs, columns=[f'item_emb_{i}' for i in range(emb_dim)])
    item_emb_df.insert(0, 'item_id', item_ids)
    user_emb_df.to_parquet(
        os.path.join(paths['feature_store'], 'user_embeddings.parquet'),
        index=False, compression='snappy')
    item_emb_df.to_parquet(
        os.path.join(paths['feature_store'], 'item_embeddings.parquet'),
        index=False, compression='snappy')

    # ── 完成报告 ──
    pop_r20 = baselines['popular'].get('Recall@20', 0)
    model_r20 = metrics.get('Recall@20', 0)
    lift = (model_r20 - pop_r20) / pop_r20 * 100 if pop_r20 > 0 else 0
    log.info("=" * 50)
    log.info("✅  Step2 Pipeline 完成！")
    log.info(f"  FAISS 索引:    {fi.index.ntotal:,} 个视频向量")
    log.info(f"  Recall@20:     {model_r20:.4f}")
    log.info(f"  Hit@20:        {metrics.get('Hit@20', 0):.4f}")
    log.info(f"  NDCG@20:       {metrics.get('NDCG@20', 0):.4f}")
    log.info(f"  vs 热门召回:   {lift:+.1f}%")


if __name__ == '__main__':
    main()
