"""
Train patch-cell-graph GNN with a choice of graph backbone + aggregation method.

Uses a PATIENT-GROUPED, label-stratified split (see splits.py) so no patient's
slides straddle train/val/test — avoiding data leakage.

Examples:
    python3 recurrence_gnn/train.py --aggregator attention --gnn_type gat
    python3 recurrence_gnn/train.py --aggregator mean      --gnn_type gin --hidden 128

Outputs (under results/runs/<aggregator>_<gnn_type>/):
    best.pt         best-val-AUC checkpoint
    train.log       full text log
    history.csv     per-epoch metrics
    curves.png      loss/acc/AUC vs epoch
    confusion.png   test-set confusion matrix
    predictions.csv per-slide test predictions

`train_one(cfg)` is importable (used by sweep.py) and returns a results dict.
"""
import os
import csv
import copy
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from config import Config
from dataset import SlideGraphDataset, collate_slides, _patient_barcode
from model import RecurrenceModel
from splits import grouped_stratified_split, split_summary
import plot_utils

try:
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def auc_score(y_true, y_prob):
    if _HAS_SK and len(set(y_true)) > 1:
        return roc_auc_score(y_true, y_prob)
    return float("nan")


def run_epoch(model, loader, device, criterion, optimizer=None, return_preds=False):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_loss, ys, ps, sids = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for batch, patch_to_slide, y, slide_ids in loader:
            batch          = batch.to(device)
            patch_to_slide = patch_to_slide.to(device)
            y              = y.to(device)

            logit = model(batch, patch_to_slide, num_slides=y.numel())
            loss  = criterion(logit, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * y.numel()
            ys.extend(y.detach().cpu().tolist())
            ps.extend(torch.sigmoid(logit).detach().cpu().tolist())
            sids.extend(slide_ids)

    n   = len(ys)
    acc = float(np.mean([(p >= 0.5) == (t >= 0.5) for p, t in zip(ps, ys)]))
    metrics = (total_loss / n, acc, auc_score(ys, ps))
    if return_preds:
        return metrics, ys, ps, sids
    return metrics


def train_one(cfg, verbose=True):
    """Train a single configuration. Returns a results dict with best val + test
    metrics. Writes checkpoint, logs, history CSV, curve + confusion-matrix PNGs."""
    set_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    run_name = f"{cfg.aggregator}_{cfg.gnn_type}"
    out_dir  = os.path.join(cfg.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    log_file = open(os.path.join(out_dir, "train.log"), "w", buffering=1)

    def log(msg=""):
        if verbose:
            print(msg)
        log_file.write(str(msg) + "\n")

    # ── dataset ──────────────────────────────────────────────────────────────
    full = SlideGraphDataset(cfg.graph_dir, cfg.labels_csv,
                             cfg.max_patches_per_slide, train=False)
    log(f"\n[{run_name}] Dataset: {full.summary()}")
    in_dim = int(cfg.in_dim) if cfg.in_dim is not None else full.feature_dim()
    log(f"  node feature dim = {in_dim}  |  device = {device}")

    # ── patient-grouped split (no leakage) ───────────────────────────────────
    tr_idx, va_idx, te_idx = grouped_stratified_split(
        full.items, cfg.val_frac, cfg.test_frac, cfg.seed
    )
    log("\n" + split_summary(full.items, tr_idx, va_idx, te_idx) + "\n")

    train_ds = copy.copy(full)
    train_ds.train = True
    train_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=cfg.batch_size,
                              shuffle=True,  collate_fn=collate_slides,
                              num_workers=cfg.num_workers)
    val_loader   = DataLoader(Subset(full,     va_idx), batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=collate_slides,
                              num_workers=cfg.num_workers)
    test_loader  = DataLoader(Subset(full,     te_idx), batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=collate_slides,
                              num_workers=cfg.num_workers) if te_idx else None

    # ── model ────────────────────────────────────────────────────────────────
    model = RecurrenceModel(
        in_dim=in_dim, hidden=cfg.hidden, gnn_layers=cfg.gnn_layers,
        gnn_type=cfg.gnn_type, aggregator=cfg.aggregator, dropout=cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  parameters: {n_params:,}\n")

    # class imbalance → weight positive class
    tr_labels  = [full.labels[i] for i in tr_idx]
    n_pos      = max(1, sum(tr_labels))
    n_neg      = max(1, len(tr_labels) - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    # ── training loop ────────────────────────────────────────────────────────
    best_auc, best_epoch = -1.0, 0
    best_path  = os.path.join(out_dir, cfg.ckpt_name)
    history    = []

    log(f"{'ep':>4}  {'tr_loss':>8} {'tr_acc':>7} {'tr_auc':>7}  "
        f"{'va_loss':>8} {'va_acc':>7} {'va_auc':>7}")
    log("-" * 62)

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, device, criterion, optimizer)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader,   device, criterion)
        scheduler.step()

        history.append(dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc, tr_auc=tr_auc,
                            va_loss=va_loss, va_acc=va_acc, va_auc=va_auc))

        flag = ""
        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc, best_epoch = va_auc, epoch
            torch.save({
                "model_state": model.state_dict(), "in_dim": in_dim,
                "hidden": cfg.hidden, "gnn_layers": cfg.gnn_layers,
                "gnn_type": cfg.gnn_type, "aggregator": cfg.aggregator,
                "dropout": cfg.dropout, "val_auc": va_auc, "epoch": epoch,
            }, best_path)
            flag = "  <-best"

        log(f"{epoch:4d}  {tr_loss:8.4f} {tr_acc:7.3f} {tr_auc:7.3f}  "
            f"{va_loss:8.4f} {va_acc:7.3f} {va_auc:7.3f}{flag}")

    log(f"\nbest val AUC = {best_auc:.4f} @ epoch {best_epoch}  ->  {best_path}")

    # ── write history CSV + curve plot ───────────────────────────────────────
    hist_csv = os.path.join(out_dir, "history.csv")
    with open(hist_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)
    try:
        plot_utils.plot_training_curves(hist_csv, os.path.join(out_dir, "curves.png"),
                                        title=f"{run_name} (h={cfg.hidden}, lr={cfg.lr})")
    except Exception as e:
        log(f"  [warn] curve plot failed: {e}")

    # ── test evaluation + confusion matrix ───────────────────────────────────
    result = dict(gnn_type=cfg.gnn_type, aggregator=cfg.aggregator,
                  hidden=cfg.hidden, lr=cfg.lr, gnn_layers=cfg.gnn_layers,
                  dropout=cfg.dropout, n_params=n_params,
                  best_epoch=best_epoch, val_auc=best_auc,
                  test_loss=float("nan"), test_acc=float("nan"), test_auc=float("nan"))

    if test_loader:
        ckpt = torch.load(best_path, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        (te_loss, te_acc, te_auc), ys, ps, sids = run_epoch(
            model, test_loader, device, criterion, return_preds=True)
        result.update(test_loss=te_loss, test_acc=te_acc, test_auc=te_auc)
        log(f"\ntest  loss {te_loss:.4f}  acc {te_acc:.3f}  AUC {te_auc:.4f}")

        preds = [1 if p >= 0.5 else 0 for p in ps]
        if _HAS_SK and len(set(ys)) > 1:
            cm = confusion_matrix(ys, preds, labels=[0, 1])
            log("\nConfusion matrix (rows=true, cols=pred):")
            log(f"          pred_L  pred_H")
            log(f"  true_L  {cm[0,0]:6d}  {cm[0,1]:6d}")
            log(f"  true_H  {cm[1,0]:6d}  {cm[1,1]:6d}")
            log("\n" + classification_report(ys, preds,
                        target_names=["Low", "High"], zero_division=0))
            try:
                plot_utils.plot_confusion_matrix(
                    cm, os.path.join(out_dir, "confusion.png"),
                    title=f"{run_name}  (test AUC={te_auc:.3f})")
            except Exception as e:
                log(f"  [warn] confusion plot failed: {e}")

        # per-slide predictions
        with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["slide_id", "patient_id", "score", "pred", "label"])
            for sid, score, pred, lab in zip(sids, ps, preds, ys):
                w.writerow([sid, _patient_barcode(sid), f"{score:.4f}", pred, int(lab)])

    log_file.close()
    return result


def main():
    cfg = Config()
    ap  = argparse.ArgumentParser()
    for k, v in vars(cfg).items():
        ap.add_argument(f"--{k}", type=type(v) if v is not None else str, default=v)
    args = ap.parse_args()
    cfg  = Config(**vars(args))
    train_one(cfg, verbose=True)


if __name__ == "__main__":
    main()
