# serving/serving_api.py
# FastAPI 推荐服务 — KuaiRec 版本
# ============================================================
#
# 接口:
#   GET /recommend/{user_id}          → 获取推荐列表
#   GET /recommend/{user_id}?top_n=20 → 指定返回数量
#   GET /health                        → 健康检查
#   GET /stats                         → 服务统计信息
#
# 本地运行:
#   uvicorn serving.serving_api:app --reload --port 8000
#
# ============================================================

import os
import time
import logging
import pickle
import numpy as np
import pandas as pd
import faiss
import torch
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.preprocessing import MinMaxScaler

# 项目模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.ranking import DINMultiTask

logging.basicConfig(level=logging.INFO)
log = logging.getLogger('ServingAPI')

# ── 全局状态 ──
_state = {}


# ══════════════════════════════════════════════
# 启动时加载所有模型和数据
# ══════════════════════════════════════════════
def load_all(base_dir: str):
    """加载推理所需的所有资源"""
    paths = {
        'features'     : os.path.join(base_dir, 'data', 'features'),
        'feature_store': os.path.join(base_dir, 'feature_store'),
        'recall_models': os.path.join(base_dir, 'models', 'recall'),
        'rank_models'  : os.path.join(base_dir, 'models', 'ranking'),
    }

    log.info("加载特征数据...")
    user_feat   = pd.read_parquet(os.path.join(paths['features'],  'user_features.parquet'))
    item_feat   = pd.read_parquet(os.path.join(paths['features'],  'item_features.parquet'))
    seq_df      = pd.read_parquet(os.path.join(paths['features'],  'user_sequences.parquet'))
    user_emb_df = pd.read_parquet(os.path.join(paths['feature_store'], 'user_embeddings.parquet'))
    item_emb_df = pd.read_parquet(os.path.join(paths['feature_store'], 'item_embeddings.parquet'))

    log.info("加载 ID 映射...")
    with open(os.path.join(paths['recall_models'], 'id_mappings.pkl'), 'rb') as f:
        id_maps = pickle.load(f)
    user2idx = id_maps['user2idx']
    item2idx = id_maps['item2idx']

    log.info("加载特征 Scaler...")
    with open(os.path.join(paths['rank_models'], 'scalers.pkl'), 'rb') as f:
        scalers = pickle.load(f)

    # 特征归一化
    USER_COLS = [c for c in user_feat.columns
                 if c != 'user_id' and user_feat[c].dtype in
                 [np.float64, np.float32, np.int64, np.int32]]
    ITEM_COLS = [c for c in item_feat.columns
                 if c != 'item_id' and item_feat[c].dtype in
                 [np.float64, np.float32, np.int64, np.int32]]

    u_np = scalers['user'].transform(
        user_feat[USER_COLS].fillna(0).values.astype(np.float32))
    i_np = scalers['item'].transform(
        item_feat[ITEM_COLS].fillna(0).values.astype(np.float32))

    user_feat_dict = {uid: u_np[i] for i, uid in enumerate(user_feat['user_id'])}
    item_feat_dict = {iid: i_np[i] for i, iid in enumerate(item_feat['item_id'])}

    # Embedding 字典
    EMB_COLS_U    = [c for c in user_emb_df.columns if c.startswith('user_emb_')]
    EMB_COLS_I    = [c for c in item_emb_df.columns if c.startswith('item_emb_')]
    user_emb_dict = {row['user_id']: row[EMB_COLS_U].values.astype(np.float32)
                     for _, row in user_emb_df.iterrows()}
    item_emb_dict = {row['item_id']: row[EMB_COLS_I].values.astype(np.float32)
                     for _, row in item_emb_df.iterrows()}

    # 序列字典
    seq_dict = {}
    for _, row in seq_df.iterrows():
        raw = str(row.get('pos_seq', '') or '')
        ids = [item2idx[int(x)] for x in raw.split(',')
               if x and int(x) in item2idx]
        seq_dict[row['user_id']] = ids

    log.info("加载 FAISS 索引...")
    faiss_index     = faiss.read_index(
        os.path.join(paths['recall_models'], 'faiss_item.index'))
    recall_item_ids = np.load(
        os.path.join(paths['recall_models'], 'item_ids.npy')).tolist()
    recall_user_ids = np.load(
        os.path.join(paths['recall_models'], 'user_ids.npy')).tolist()
    user_id2row     = {uid: i for i, uid in enumerate(recall_user_ids)}

    EMB_DIM = len(EMB_COLS_U)
    user_emb_matrix = np.stack([
        user_emb_dict.get(uid, np.zeros(EMB_DIM, np.float32))
        for uid in recall_user_ids
    ]).astype(np.float32)
    faiss.normalize_L2(user_emb_matrix)

    log.info("加载 DIN 精排模型...")
    ckpt = torch.load(
        os.path.join(paths['rank_models'], 'din_best.pt'),
        map_location='cpu', weights_only=False)
    cfg  = ckpt['cfg']
    din_model = DINMultiTask(
        n_items       = len(item2idx),
        i_feat_dim    = len(ITEM_COLS),
        u_feat_dim    = len(USER_COLS),
        tower_emb_dim = EMB_DIM,
        cfg           = cfg,
    )
    din_model.load_state_dict(ckpt['model'])
    din_model.eval()

    log.info("✅ 所有资源加载完成")
    return {
        'user_feat_dict' : user_feat_dict,
        'item_feat_dict' : item_feat_dict,
        'user_emb_dict'  : user_emb_dict,
        'item_emb_dict'  : item_emb_dict,
        'seq_dict'       : seq_dict,
        'faiss_index'    : faiss_index,
        'recall_item_ids': recall_item_ids,
        'user_id2row'    : user_id2row,
        'din_model'      : din_model,
        'user2idx'       : user2idx,
        'item2idx'       : item2idx,
        'cfg'            : cfg,
        'u_dim'          : len(USER_COLS),
        'i_dim'          : len(ITEM_COLS),
        'emb_dim'        : EMB_DIM,
        'user_emb_matrix': user_emb_matrix,
        'start_time'     : time.time(),
        'request_count'  : 0,
    }


# ══════════════════════════════════════════════
# 推理函数
# ══════════════════════════════════════════════
def recommend(user_id: int, recall_k: int = 200,
              top_n: int = 10, alpha: float = 0.5):
    s = _state
    if user_id not in s['user_id2row']:
        return None

    # 阶段1: 双塔召回
    u_vec = s['user_emb_matrix'][s['user_id2row'][user_id]:s['user_id2row'][user_id]+1]
    _, idxs   = s['faiss_index'].search(u_vec, recall_k)
    cand_ids  = [int(s['recall_item_ids'][i]) for i in idxs[0]]

    # 阶段2: DIN 精排
    cfg     = s['cfg']
    u_dim   = s['u_dim']
    i_dim   = s['i_dim']
    emb_dim = s['emb_dim']
    seq_len = cfg['seq_len']
    device  = 'cpu'

    u_feat_t = torch.tensor(
        s['user_feat_dict'].get(user_id, np.zeros(u_dim, np.float32)),
        dtype=torch.float).unsqueeze(0)
    u_emb_t  = torch.tensor(
        s['user_emb_dict'].get(user_id, np.zeros(emb_dim, np.float32)),
        dtype=torch.float).unsqueeze(0)

    seq     = s['seq_dict'].get(user_id, [])[-seq_len:]
    pad_len = seq_len - len(seq)
    seq_t   = torch.tensor([0]*pad_len + seq, dtype=torch.long).unsqueeze(0)
    mask_t  = torch.tensor([0]*pad_len + [1]*len(seq), dtype=torch.float).unsqueeze(0)

    scores = []
    batch  = 64
    with torch.no_grad():
        for start in range(0, len(cand_ids), batch):
            b_iids = cand_ids[start:start+batch]
            B = len(b_iids)
            i_feat_b = torch.tensor(
                np.stack([s['item_feat_dict'].get(i, np.zeros(i_dim, np.float32))
                          for i in b_iids]), dtype=torch.float)
            i_emb_b  = torch.tensor(
                np.stack([s['item_emb_dict'].get(i, np.zeros(emb_dim, np.float32))
                          for i in b_iids]), dtype=torch.float)
            ctr_pred, dur_pred = s['din_model'](
                u_feat_t.expand(B,-1), u_emb_t.expand(B,-1),
                i_feat_b, i_emb_b,
                seq_t.expand(B,-1), mask_t.expand(B,-1))
            final = alpha * ctr_pred + (1-alpha) * dur_pred
            for iid, ctr, dur, fs in zip(
                b_iids,
                ctr_pred.numpy(), dur_pred.numpy(), final.numpy()):
                scores.append({
                    'item_id'       : int(iid),
                    'ctr_score'     : round(float(ctr), 4),
                    'duration_score': round(float(dur), 4),
                    'final_score'   : round(float(fs),  4),
                })

    # 阶段3: 排序取 Top N
    scores.sort(key=lambda x: x['final_score'], reverse=True)
    return scores[:top_n]


# ══════════════════════════════════════════════
# FastAPI 应用
# ══════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    base_dir   = os.environ.get('BASE_DIR', '/app/model_artifacts')
    gcs_bucket = os.environ.get('GCS_BUCKET', '')
    gcs_prefix = os.environ.get('GCS_PREFIX', 'VideoRecSys/model_artifacts')

    if gcs_bucket:
        log.info(f"从 GCS 下载模型文件  bucket={gcs_bucket}")
        from serving.gcs_loader import download_model_artifacts
        download_model_artifacts(gcs_bucket, gcs_prefix, base_dir)

    log.info(f"加载资源  BASE_DIR={base_dir}")
    _state.update(load_all(base_dir))
    yield
    _state.clear()

app = FastAPI(
    title       = '🎬 VideoRecSys API',
    description = 'KuaiRec 短视频推荐系统 — 双塔召回 + DIN 多任务精排',
    version     = '1.0.0',
    lifespan    = lifespan,
)

# 允许跨域（方便前端调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


# ── 数据模型 ──
class RecommendItem(BaseModel):
    item_id:        int
    ctr_score:      float
    duration_score: float
    final_score:    float

class RecommendResponse(BaseModel):
    user_id:       int
    recommendations: List[RecommendItem]
    recall_k:      int
    top_n:         int
    latency_ms:    float


# ── 接口 ──
@app.get('/health')
def health():
    """健康检查（Cloud Run 需要这个接口）"""
    return {'status': 'ok', 'model': 'DIN MultiTask', 'dataset': 'KuaiRec'}


@app.get('/stats')
def stats():
    """服务统计信息"""
    uptime = time.time() - _state.get('start_time', time.time())
    return {
        'uptime_seconds' : round(uptime, 1),
        'request_count'  : _state.get('request_count', 0),
        'user_count'     : len(_state.get('user2idx', {})),
        'item_count'     : len(_state.get('item2idx', {})),
        'model'          : 'DIN MultiTask (CTR + WatchRatio)',
        'dataset'        : 'KuaiRec small_matrix',
    }


@app.get('/recommend/{user_id}', response_model=RecommendResponse)
def get_recommendations(
    user_id:  int,
    top_n:    int   = Query(default=10, ge=1, le=50),
    recall_k: int   = Query(default=200, ge=10, le=500),
    alpha:    float = Query(default=0.5, ge=0.0, le=1.0),
):
    """
    获取用户推荐列表

    - **user_id**: 用户 ID
    - **top_n**: 返回推荐数量（1~50，默认10）
    - **recall_k**: 召回候选数（10~500，默认200）
    - **alpha**: CTR 权重（0~1，默认0.5，即 0.5*CTR + 0.5*完播率）
    """
    t0 = time.time()
    _state['request_count'] = _state.get('request_count', 0) + 1

    if not _state:
        raise HTTPException(status_code=503, detail='模型未加载')

    results = recommend(user_id, recall_k, top_n, alpha)

    if results is None:
        raise HTTPException(
            status_code=404,
            detail=f'用户 {user_id} 不在系统中（冷启动用户）')

    latency = (time.time() - t0) * 1000
    return RecommendResponse(
        user_id         = user_id,
        recommendations = results,
        recall_k        = recall_k,
        top_n           = top_n,
        latency_ms      = round(latency, 2),
    )


@app.get('/users')
def list_users(limit: int = Query(default=20, le=100)):
    """列出系统中的用户（方便测试）"""
    users = list(_state.get('user2idx', {}).keys())[:limit]
    return {'users': users, 'total': len(_state.get('user2idx', {}))}
