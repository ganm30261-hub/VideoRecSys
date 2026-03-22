# models/recall/recall_trainer.py
# 双塔模型训练器 — KuaiRec 版本
# 对应 Step2 Cell 6 + Cell 8
# ============================================================

import os
import time
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
from tqdm import tqdm

from .two_tower_model      import TwoTowerModel
from .weighted_infonce_loss import WeightedInfoNCELoss
from .faiss_index           import FaissIndex

logger = logging.getLogger(__name__)


class RecallTrainer:
    """
    双塔召回模型训练器

    功能:
        - 训练循环（含 Early Stopping）
        - 全量 Embedding 生成
        - FAISS 索引构建
        - 模型保存/加载
    """

    def __init__(self, model: TwoTowerModel, cfg: dict, save_dir: str):
        self.model    = model
        self.cfg      = cfg
        self.save_dir = save_dir
        self.device   = cfg['device']
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = WeightedInfoNCELoss(temperature=cfg['temperature'])
        self.optimizer = torch.optim.Adam(
            list(model.user_tower.parameters()) +
            list(model.item_tower.parameters()),
            lr=cfg['lr'], weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg['epochs'])

        self.history: Dict[str, List] = {
            'train_loss': [], 'val_loss': [], 'lr': []}
        self.best_loss  = float('inf')
        self.no_improve = 0

    # ─────────────────────────────────────────
    # 训练
    # ─────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        """
        执行完整训练循环

        Returns:
            训练历史 {'train_loss': [...], 'val_loss': [...], 'lr': [...]}
        """
        logger.info(
            f"开始训练  epochs={self.cfg['epochs']}  "
            f"batch={self.cfg['batch_size']}  device={self.device}"
        )

        for epoch in range(1, self.cfg['epochs'] + 1):
            t0        = time.time()
            tr_loss   = self._run_epoch(train_loader, training=True)
            va_loss   = self._run_epoch(val_loader,   training=False)
            cur_lr    = self.optimizer.param_groups[0]['lr']
            self.scheduler.step()

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(va_loss)
            self.history['lr'].append(cur_lr)

            flag = '⬆ best' if va_loss < self.best_loss else ''
            print(
                f"Epoch {epoch:3d}/{self.cfg['epochs']}  "
                f"loss={tr_loss:.4f}/{va_loss:.4f}  "
                f"lr={cur_lr:.2e}  "
                f"{time.time()-t0:.1f}s  {flag}"
            )

            if va_loss < self.best_loss:
                self.best_loss  = va_loss
                self.no_improve = 0
                self._save_checkpoint(epoch, va_loss)
            else:
                self.no_improve += 1
                if self.no_improve >= self.cfg['early_stop']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        self._load_best_checkpoint()
        logger.info(f"训练完成  best_val_loss={self.best_loss:.4f}")
        return self.history

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.user_tower.train(training)
        self.model.item_tower.train(training)
        total_loss = 0.0
        ctx = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for batch in loader:
                u_idx, i_idx, u_feat, i_feat, seq, weight = [
                    b.to(self.device) for b in batch]
                u_emb = self.model.encode_user(u_idx, u_feat, seq)
                i_emb = self.model.encode_item(i_idx, i_feat)
                loss  = self.criterion(u_emb, i_emb, weight)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.user_tower.parameters()) +
                        list(self.model.item_tower.parameters()), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()

        return total_loss / len(loader)

    # ─────────────────────────────────────────
    # Embedding 生成
    # ─────────────────────────────────────────

    @torch.no_grad()
    def generate_embeddings(
        self,
        user_ids:        List,
        item_ids:        List,
        user_feat_dict:  Dict,
        item_feat_dict:  Dict,
        seq_dict:        Dict,
        user2idx:        Dict,
        item2idx:        Dict,
        batch_size:      int = 512,
    ) -> tuple:
        """
        生成全量 User / Item Embedding

        Returns:
            (user_embs, item_embs): np.ndarray [N_users, D], [N_items, D]
        """
        self.model.user_tower.eval()
        self.model.item_tower.eval()
        seq_len  = self.cfg['seq_len']
        u_dim    = len(next(iter(user_feat_dict.values())))
        i_dim    = len(next(iter(item_feat_dict.values())))

        # 用户 Embedding
        logger.info("生成用户 Embedding...")
        user_embs = []
        for start in tqdm(range(0, len(user_ids), batch_size), desc='User Emb'):
            batch_uids = user_ids[start:start + batch_size]
            idx_t  = torch.tensor(
                [user2idx[u] for u in batch_uids],
                dtype=torch.long, device=self.device)
            feat_t = torch.tensor(
                np.stack([user_feat_dict.get(u, np.zeros(u_dim, np.float32))
                          for u in batch_uids]),
                dtype=torch.float, device=self.device)
            seqs = []
            for uid in batch_uids:
                s = seq_dict.get(uid, [])[-seq_len:]
                seqs.append([0] * (seq_len - len(s)) + s)
            seq_t = torch.tensor(seqs, dtype=torch.long, device=self.device)
            emb   = self.model.encode_user(idx_t, feat_t, seq_t)
            user_embs.append(emb.cpu().numpy())

        # 视频 Embedding
        logger.info("生成视频 Embedding...")
        item_embs = []
        for start in tqdm(range(0, len(item_ids), batch_size), desc='Item Emb'):
            batch_iids = item_ids[start:start + batch_size]
            idx_t  = torch.tensor(
                [item2idx[i] for i in batch_iids],
                dtype=torch.long, device=self.device)
            feat_t = torch.tensor(
                np.stack([item_feat_dict.get(i, np.zeros(i_dim, np.float32))
                          for i in batch_iids]),
                dtype=torch.float, device=self.device)
            emb = self.model.encode_item(idx_t, feat_t)
            item_embs.append(emb.cpu().numpy())

        return np.vstack(user_embs), np.vstack(item_embs)

    # ─────────────────────────────────────────
    # 模型保存/加载
    # ─────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        torch.save(self.model.user_tower.state_dict(),
                   os.path.join(self.save_dir, 'user_tower.pt'))
        torch.save(self.model.item_tower.state_dict(),
                   os.path.join(self.save_dir, 'item_tower.pt'))

    def _load_best_checkpoint(self) -> None:
        self.model.user_tower.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'user_tower.pt'),
                       map_location=self.device, weights_only=True))
        self.model.item_tower.load_state_dict(
            torch.load(os.path.join(self.save_dir, 'item_tower.pt'),
                       map_location=self.device, weights_only=True))
        logger.info("最优权重加载完成")

    def save_id_mappings(self, user2idx, item2idx, idx2user, idx2item) -> None:
        path = os.path.join(self.save_dir, 'id_mappings.pkl')
        with open(path, 'wb') as f:
            pickle.dump({
                'user2idx': user2idx, 'item2idx': item2idx,
                'idx2user': idx2user, 'idx2item': idx2item,
            }, f)
        logger.info(f"ID 映射保存 → {path}")
