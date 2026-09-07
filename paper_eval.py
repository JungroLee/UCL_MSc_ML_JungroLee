"""
Paper-matched evaluation (Howard et al. 2023), unified for classification & regression.

Outputs -> results/runs/paper_eval/<task>_<agg>_<gnn>_h{h}_lr{lr}/
    fold{k}_curves.png, fold{k}_history.csv
    confusion.png, predictions.csv, result.txt

Usage
-----
    python3 recurrence_gnn/paper_eval.py --task classification \
        --gnn_type gat --aggregator slide_gnn --hidden 256 --lr 1e-3
    python3 recurrence_gnn/paper_eval.py --task regression \
        --gnn_type gat --aggregator attention --hidden 256 --lr 1e-4 \
        --epochs 200 --dropout 0.1
"""
import os
import csv
import copy
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import SlideGraphDataset, collate_slides, _patient_barcode
from model import RecurrenceModel
from train import set_seed, auc_score
from hrher2 import hrher2_slide_stems
import plot_utils
from paths import DATA, RUNS

try:
    from sklearn.metrics import (confusion_matrix, classification_report,
                                 r2_score)
    from scipy.stats import pearsonr, spearmanr
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


# ── data helpers ──────────────────────────────────────────────────────────────

def load_folds(clinical_csv):
    df = pd.read_csv(clinical_csv).dropna(subset=["CV3_odx85_mip"])
    return {str(s): int(f) for s, f in zip(df["slide"], df["CV3_odx85_mip"])}


def load_continuous(clinical_csv):
    df = pd.read_csv(clinical_csv)
    return {str(s): float(v) for s, v in zip(df["slide"], df["odx_train"])
            if pd.notna(v)}


def to_patient(sids, ys, scores, outs):
    """Aggregate slide-level predictions to PATIENT level (mean over a patient's
    slides), matching the paper's patient-level AUROC (n=535). A patient's binary
    label is shared across their slides."""
    by_pat = defaultdict(lambda: {"y": None, "sc": [], "out": []})
    for s, y, sc, o in zip(sids, ys, scores, outs):
        p = _patient_barcode(s)
        by_pat[p]["y"] = y
        by_pat[p]["sc"].append(sc)
        by_pat[p]["out"].append(o)
    pids = sorted(by_pat)
    py  = [by_pat[p]["y"] for p in pids]
    psc = [float(np.mean(by_pat[p]["sc"])) for p in pids]
    pout= [float(np.mean(by_pat[p]["out"])) for p in pids]
    return pids, py, psc, pout


def carve_val(pool_idx, items, val_frac, seed):
    """Patient-grouped val slice out of a set of item indices."""
    by_pat = defaultdict(list)
    for i in pool_idx:
        by_pat[_patient_barcode(items[i][0])].append(i)
    pats = sorted(by_pat)
    rng = np.random.RandomState(seed)
    rng.shuffle(pats)
    n_val = max(1, int(len(pats) * val_frac))
    val_pats = set(pats[:n_val])
    tr, va = [], []
    for p in pats:
        (va if p in val_pats else tr).extend(by_pat[p])
    return tr, va


# ── epoch runner (task-aware) ─────────────────────────────────────────────────

def run_epoch(model, loader, device, criterion, task, cont_map,
              optimizer=None, return_preds=False):
    train = optimizer is not None
    model.train() if train else model.eval()
    tot, ys, outs, sids = 0.0, [], [], []

    with torch.set_grad_enabled(train):
        for batch, p2s, y, slide_ids in loader:
            batch = batch.to(device); p2s = p2s.to(device)
            if task == "regression":
                target = torch.tensor([cont_map[s] for s in slide_ids],
                                      dtype=torch.float, device=device)
            else:
                target = y.to(device)
            out = model(batch, p2s, num_slides=y.numel())
            loss = criterion(out, target)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            tot += loss.item() * y.numel()
            ys.extend(y.tolist())                         # binary label (for AUROC)
            outs.extend(out.detach().cpu().tolist())      # logits or continuous pred
            sids.extend(slide_ids)

    n = len(ys)
    # score for AUROC: sigmoid(logit) for classification, raw pred for regression
    if task == "classification":
        scores = [1 / (1 + np.exp(-o)) for o in outs]
    else:
        scores = outs
    auc = auc_score(ys, scores)
    if return_preds:
        return tot / n, auc, ys, outs, scores, sids
    return tot / n, auc


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["classification", "regression"],
                    default="classification")
    ap.add_argument("--graph_dir",    default=f"{DATA}/Another_dataset/graphs_v2/feat_morph")
    ap.add_argument("--labels_csv",   default=f"{DATA}/labels.csv")
    ap.add_argument("--clinical_csv", default=f"{DATA}/tcga_brca_complete.csv")
    ap.add_argument("--gnn_type",     default="gat")
    ap.add_argument("--aggregator",   default="slide_gnn")
    ap.add_argument("--hidden",       type=int,   default=256)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--gnn_layers",   type=int,   default=3)
    ap.add_argument("--dropout",      type=float, default=0.25)
    ap.add_argument("--epochs",       type=int,   default=60)
    ap.add_argument("--batch_size",   type=int,   default=2)
    ap.add_argument("--max_patches",  type=int,   default=32)
    ap.add_argument("--num_workers",  type=int,   default=4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--val_frac",     type=float, default=0.15)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--out_dir",      default=f"{RUNS}/paper_eval")
    ap.add_argument("--device",       default="cuda")
    ap.add_argument("--fm_emb",       default="",
                    help="patch-level foundation-model embedding cache (.pt) to fuse")
    ap.add_argument("--img_mode",     default="concat",
                    choices=["concat", "only", "gated"],
                    help="'only' = image-only ablation (ignores the cell graph)")
    ap.add_argument("--tag_suffix",   default="",
                    help="appended to the run directory name, e.g. _fm or _imgonly")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    tag = (f"{args.task}_{args.aggregator}_{args.gnn_type}"
           f"_h{args.hidden}_lr{args.lr:g}{args.tag_suffix}")
    out_dir = os.path.join(args.out_dir, tag)
    os.makedirs(out_dir, exist_ok=True)
    log_lines = [f"# Paper-matched 3-fold CV (HR+/HER2- eval) — {tag}\n"]

    def log(m=""):
        print(m); log_lines.append(str(m))

    full = SlideGraphDataset(args.graph_dir, args.labels_csv,
                             fm_emb=args.fm_emb or None)
    in_dim = full.feature_dim()
    img_dim = full.img_dim()
    fold_of = load_folds(args.clinical_csv)
    cont_map = load_continuous(args.clinical_csv) if args.task == "regression" else {}
    hrher2 = hrher2_slide_stems(args.clinical_csv)

    item_fold = [fold_of.get(stem) for stem, _, _ in full.items]
    is_hrher2 = [stem in hrher2 for stem, _, _ in full.items]
    folds = sorted({f for f in item_fold if f is not None})

    log(f"task={args.task}  config={args.aggregator}/{args.gnn_type} "
        f"h{args.hidden} lr{args.lr:g}  node_dim={in_dim}  device={device}")
    if img_dim:
        log(f"fusion: patch image embedding dim={img_dim} mode={args.img_mode} "
            f"({args.fm_emb})")
    log(f"folds present: {folds}")
    log(f"HR+/HER2- slides in dataset: {sum(is_hrher2)}  "
        f"(eval restricted to these)\n")

    all_y, all_score, all_out, all_sid = [], [], [], []
    fold_aucs = []
    img_gates = []          # gated fusion only: learned image:graph weight per fold

    for k in folds:
        te_idx = [i for i, f in enumerate(item_fold) if f == k]
        pool   = [i for i, f in enumerate(item_fold) if f is not None and f != k]
        tr_idx, va_idx = carve_val(pool, full.items, args.val_frac, args.seed)

        set_seed(args.seed)
        train_ds = copy.copy(full); train_ds.train = True
        train_ds.max_patches = args.max_patches
        tr_loader = DataLoader(Subset(train_ds, tr_idx), batch_size=args.batch_size,
                               shuffle=True, collate_fn=collate_slides,
                               num_workers=args.num_workers)
        va_loader = DataLoader(Subset(full, va_idx), batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_slides,
                               num_workers=args.num_workers)
        te_loader = DataLoader(Subset(full, te_idx), batch_size=args.batch_size,
                               shuffle=False, collate_fn=collate_slides,
                               num_workers=args.num_workers)

        model = RecurrenceModel(in_dim=in_dim, hidden=args.hidden,
                                gnn_layers=args.gnn_layers, gnn_type=args.gnn_type,
                                aggregator=args.aggregator, dropout=args.dropout,
                                img_dim=img_dim, img_mode=args.img_mode).to(device)

        if args.task == "classification":
            tr_lab = [full.labels[i] for i in tr_idx]
            npos = max(1, sum(tr_lab)); nneg = max(1, len(tr_lab) - npos)
            criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([nneg / npos], device=device))
        else:
            criterion = nn.MSELoss()

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.epochs, eta_min=args.lr * 0.01)

        # ---- train, tracking history + best (val AUROC on HR+/HER2- subset) ----
        # model selection uses the HR+/HER2- val subset to match the paper's
        # reported population; one val pass per epoch (kept small & fast).
        hist = []
        best_auc, best_state = -1.0, None
        hrher2_va = [i for i in va_idx if is_hrher2[i]] or va_idx
        va_hr_loader = DataLoader(Subset(full, hrher2_va), batch_size=args.batch_size,
                                  shuffle=False, collate_fn=collate_slides,
                                  num_workers=args.num_workers)
        for ep in range(1, args.epochs + 1):
            tr_loss, tr_auc = run_epoch(model, tr_loader, device, criterion,
                                        args.task, cont_map, opt)
            va_loss, va_auc = run_epoch(model, va_hr_loader, device, criterion,
                                        args.task, cont_map)
            sched.step()
            hist.append(dict(epoch=ep, tr_loss=tr_loss, tr_acc=tr_auc, tr_auc=tr_auc,
                             va_loss=va_loss, va_acc=va_auc, va_auc=va_auc))
            if not np.isnan(va_auc) and va_auc > best_auc:
                best_auc, best_state = va_auc, copy.deepcopy(model.state_dict())

        # save curves + history
        hist_csv = os.path.join(out_dir, f"fold{k}_history.csv")
        with open(hist_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
            w.writeheader(); w.writerows(hist)
        try:
            plot_utils.plot_training_curves(hist_csv,
                os.path.join(out_dir, f"fold{k}_curves.png"),
                title=f"{tag} — fold {k}")
        except Exception as e:
            log(f"  [warn] fold{k} curve failed: {e}")

        # ---- save fold weights (for later reuse, e.g. UCH validation) ----
        torch.save({"model_state": best_state, "gnn_type": args.gnn_type,
                    "aggregator": args.aggregator, "hidden": args.hidden,
                    "gnn_layers": args.gnn_layers, "in_dim": in_dim,
                    "dropout": args.dropout, "img_dim": img_dim,
                    "img_mode": args.img_mode, "fm_emb": args.fm_emb,
                    "graph_dir": args.graph_dir,
                    "task": args.task, "fold": k, "val_auc": best_auc},
                   os.path.join(out_dir, f"best_fold{k}.pt"))

        # ---- held-out eval on HR+/HER2- subset ----
        model.load_state_dict(best_state)
        if args.img_mode == "gated":
            # image:graph weight ratio learned by this fold (see model.py).
            # >1 means the image branch dominates, <1 means the cell graph does.
            g = float(model.img_gate.detach().cpu().item())
            img_gates.append(g)
            log(f"Fold {k}: learned img_gate={g:.4f}  "
                f"({'image' if g > 1 else 'graph'}-dominated)")
        _, _, ys, outs, scores, sids = run_epoch(
            model, te_loader, device, criterion, args.task, cont_map, return_preds=True)
        keep = [j for j, s in enumerate(sids) if s in hrher2]
        y_k  = [ys[j] for j in keep]
        sc_k = [scores[j] for j in keep]
        out_k= [outs[j] for j in keep]
        sid_k= [sids[j] for j in keep]
        # aggregate slides -> patients (paper reports patient-level AUROC, n=535)
        pid_k, py_k, psc_k, pout_k = to_patient(sid_k, y_k, sc_k, out_k)
        auc_k = auc_score(py_k, psc_k)
        fold_aucs.append(auc_k)
        all_y += py_k; all_score += psc_k; all_out += pout_k; all_sid += pid_k
        log(f"Fold {k}: held-out HR+/HER2- slides={len(keep)} -> patients={len(pid_k)}  "
            f"val-best AUROC={best_auc:.4f}  ->  test AUROC(patient)={auc_k:.4f}")

        # free GPU/CPU between folds (avoid fragmentation/leak hangs on later folds)
        import gc
        del model, opt, sched, best_state, tr_loader, va_loader, te_loader, va_hr_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── aggregate ────────────────────────────────────────────────────────────
    mean_auc, std_auc = float(np.mean(fold_aucs)), float(np.std(fold_aucs))
    pooled_auc = auc_score(all_y, all_score)
    log(f"\n=== {tag} ===")
    log(f"per-fold AUROC : {['%.4f' % a for a in fold_aucs]}")
    log(f"mean AUROC     : {mean_auc:.4f} ± {std_auc:.4f}")
    log(f"pooled AUROC   : {pooled_auc:.4f}")
    log(f"paper (DL path): 0.797  (95% CI 0.680-0.901)")

    if args.task == "regression" and _HAS_SK:
        # patient-level continuous target (a patient's slides share the ODX score)
        cont_pat = {}
        for s, v in cont_map.items():
            cont_pat[_patient_barcode(s)] = v
        true = [cont_pat[p] for p in all_sid]
        r2 = r2_score(true, all_out); pr = pearsonr(true, all_out)[0]
        sp = spearmanr(true, all_out)[0]
        log(f"regression: R2 {r2:.4f}  Pearson {pr:.4f}  Spearman {sp:.4f}")

    # confusion matrix on pooled HR+/HER2- held-out
    if _HAS_SK and len(set(all_y)) > 1:
        preds = [1 if s >= 0.5 else 0 for s in all_score] if args.task == "classification" \
                else [1 if o >= 0 else 0 for o in all_out]
        cm = confusion_matrix(all_y, preds, labels=[0, 1])
        log("\nConfusion matrix (HR+/HER2- held-out, rows=true, cols=pred):")
        log(f"          pred_L  pred_H")
        log(f"  true_L  {cm[0,0]:6d}  {cm[0,1]:6d}")
        log(f"  true_H  {cm[1,0]:6d}  {cm[1,1]:6d}")
        log("\n" + classification_report(all_y, preds, target_names=["Low", "High"],
                                          zero_division=0))
        plot_utils.plot_confusion_matrix(cm, os.path.join(out_dir, "confusion.png"),
            title=f"{tag}\nHR+/HER2- held-out (mean AUROC={mean_auc:.3f})")

    with open(os.path.join(out_dir, "predictions.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["patient_id", "score", "label"])
        for p, sc, y in zip(all_sid, all_score, all_y):
            w.writerow([p, f"{sc:.4f}", int(y)])

    # ── extended binary metrics at Youden-optimal threshold + Spearman ───────
    import json
    metrics = dict(tag=tag, task=args.task, gnn_type=args.gnn_type,
                   aggregator=args.aggregator, hidden=args.hidden, lr=args.lr,
                   n_patients=len(all_y), mean_auc=mean_auc, std_auc=std_auc,
                   pooled_auc=pooled_auc)
    if img_gates:
        metrics.update(img_gate_per_fold=[round(g, 4) for g in img_gates],
                       img_gate_mean=float(np.mean(img_gates)))
        log(f"\nLearned img_gate per fold: "
            f"{', '.join(f'{g:.4f}' for g in img_gates)}  "
            f"(mean {np.mean(img_gates):.4f})")
    if _HAS_SK and len(set(all_y)) > 1:
        from sklearn.metrics import roc_curve
        yv = np.array(all_y); sv = np.array(all_score)
        fpr, tpr, thr = roc_curve(yv, sv)
        j = thr[np.argmax(tpr - fpr)]                       # Youden-optimal
        pred = (sv >= j).astype(int)
        tn, fp, fn, tp = confusion_matrix(yv, pred, labels=[0, 1]).ravel()
        metrics.update(
            threshold=float(j),
            accuracy=float((tp + tn) / len(yv)),
            recall=float(tp / (tp + fn) if tp + fn else 0),
            precision=float(tp / (tp + fp) if tp + fp else 0),
            specificity=float(tn / (tn + fp) if tn + fp else 0),
            f1=float(2 * tp / (2 * tp + fp + fn) if (2*tp+fp+fn) else 0),
            spearman=float(spearmanr(all_score, all_y)[0]),
        )
        log(f"\nAt Youden threshold {j:.3f}: acc {metrics['accuracy']:.3f}  "
            f"recall {metrics['recall']:.3f}  precision {metrics['precision']:.3f}  "
            f"specificity {metrics['specificity']:.3f}  f1 {metrics['f1']:.3f}")
        log(f"Spearman (score vs label): {metrics['spearman']:.3f}")
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(out_dir, "result.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\nSaved -> {out_dir}")
    return metrics


if __name__ == "__main__":
    main()
