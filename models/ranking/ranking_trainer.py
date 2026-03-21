# models/ranking/ranking_trainer.py
# 精排模型训练器 — KuaiRec 版本
# 对应 Step3 Cell 5
# ============================================================

import os
import time
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from typing import Dict, List, Tuple

from .multi_task_loss import MultiTaskLoss

logger = logging.getLogger(__name__)


class RankingTrainer:
    """
    精排模型训练器 — KuaiRec 版本

    功能:
        - 多任务训练循环（CTR + 完播率）
        - Early Stopping（基于 val AUC）
        - 模型保存/加载
    """

    def __init__(self, model: nn.Module, cfg: dict, save_dir: str):
        self.model    = model
        self.cfg      = cfg
        self.save_dir = save_dir
        self.device   = cfg['device']
        os.makedirs(save_dir, exist_ok=True)

        self.criterion = MultiTaskLoss(alpha=cfg['multitask_alpha'])
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=2, factor=0.5)

        self.history: Dict[str, List] = {
            'train_loss': [], 'val_loss': [],
            'train_auc':  [], 'val_auc':  [],
        }
        self.best_auc   = 0.0
        self.no_improve = 0
        self.best_epoch = 0

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
    ) -> Dict:
        """
        执行完整训练循环

        Returns:
            训练历史字典
        """
        logger.info(
            f"开始训练  epochs={self.cfg['epochs']}  "
            f"batch={self.cfg['batch_size']}  device={self.device}  "
            f"alpha={self.cfg['multitask_alpha']}"
        )

        for epoch in range(1, self.cfg['epochs'] + 1):
            t0 = time.time()
            tr_loss, tr_auc, _, _ = self._run_epoch(train_loader, training=True)
            va_loss, va_auc, _, _ = self._run_epoch(val_loader,   training=False)
            self.scheduler.step(va_auc)

            self.history['train_loss'].append(tr_loss)
            self.history['val_loss'].append(va_loss)
            self.history['train_auc'].append(tr_auc)
            self.history['val_auc'].append(va_auc)

            flag = '⬆ best' if va_auc > self.best_auc else ''
            print(
                f"Epoch {epoch:3d}/{self.cfg['epochs']}  "
                f"loss={tr_loss:.4f}/{va_loss:.4f}  "
                f"AUC={tr_auc:.4f}/{va_auc:.4f}  "
                f"{time.time()-t0:.1f}s  {flag}"
            )

            if va_auc > self.best_auc:
                self.best_auc   = va_auc
                self.best_epoch = epoch
                self.no_improve = 0
                self._save_checkpoint(epoch, va_auc)
            else:
                self.no_improve += 1
                if self.no_improve >= self.cfg['early_stop']:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        self._load_best_checkpoint()
        logger.info(
            f"训练完成  best_epoch={self.best_epoch}  "
            f"best_val_AUC={self.best_auc:.4f}"
        )
        return self.history

    def predict(self, loader: DataLoader) -> Tuple[list, list, list, list]:
        """
        推断接口，返回所有样本的预测结果

        Returns:
            ctr_preds, dur_preds, ctr_labels, dur_labels
        """
        self.model.eval()
        ctr_preds, dur_preds = [], []
        ctr_labels, dur_labels = [], []

        with torch.no_grad():
            for batch in loader:
                u_feat, u_emb, i_feat, i_emb, seq, mask, ctr_label, dur_label = [
                    b.to(self.device) for b in batch]
                ctr_pred, dur_pred = self.model(
                    u_feat, u_emb, i_feat, i_emb, seq, mask)
                ctr_preds.extend(ctr_pred.cpu().numpy())
                dur_preds.extend(dur_pred.cpu().numpy())
                ctr_labels.extend(ctr_label.cpu().numpy())
                dur_labels.extend(dur_label.cpu().numpy())

        return ctr_preds, dur_preds, ctr_labels, dur_labels

    # ─────────────────────────────────────────
    # 私有方法
    # ─────────────────────────────────────────

    def _run_epoch(
        self,
        loader:   DataLoader,
        training: bool,
    ) -> Tuple[float, float, list, list]:
        self.model.train(training)
        total_loss = 0.0
        all_preds, all_labels = [], []
        ctx = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for batch in loader:
                u_feat, u_emb, i_feat, i_emb, seq, mask, ctr_label, dur_label = [
                    b.to(self.device) for b in batch]
                ctr_pred, dur_pred = self.model(
                    u_feat, u_emb, i_feat, i_emb, seq, mask)
                loss, _, _ = self.criterion(
                    ctr_pred, dur_pred, ctr_label, dur_label)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                all_preds.extend(ctr_pred.detach().cpu().numpy())
                all_labels.extend(ctr_label.cpu().numpy())

        avg_loss = total_loss / len(loader)
        auc = roc_auc_score(all_labels, all_preds) \
              if len(set(all_labels)) > 1 else 0.5

        return avg_loss, auc, all_preds, all_labels

    def _save_checkpoint(self, epoch: int, val_auc: float) -> None:
        path = os.path.join(self.save_dir, 'din_best.pt')
        torch.save({
            'epoch'  : epoch,
            'model'  : self.model.state_dict(),
            'val_auc': val_auc,
            'cfg'    : self.cfg,
        }, path)

    def _load_best_checkpoint(self) -> None:
        path = os.path.join(self.save_dir, 'din_best.pt')
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        logger.info(
            f"最优模型加载  epoch={ckpt['epoch']}  "
            f"val_AUC={ckpt['val_auc']:.4f}"
        )
