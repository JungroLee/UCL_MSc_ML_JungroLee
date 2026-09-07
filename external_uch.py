"""
Train ONE model on ALL of TCGA (paper's external-validation protocol), then infer
on UCH and score against real clinical OncotypeDX (RS>=26).

    python3 train_full_tcga_infer_uch.py
"""
import os, csv, copy, json, argparse
from collections import defaultdict
import numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import SlideGraphDataset, collate_slides, _patient_barcode
from model import RecurrenceModel
from train import set_seed, auc_score
from paper_eval import run_epoch, load_continuous, to_patient
from paths import DATA, RUNS

try:
    from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
    from scipy.stats import spearmanr, pearsonr
    _HAS = True
except ImportError:
    _HAS = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="regression")
    ap.add_argument("--aggregator", default="attention")
    ap.add_argument("--gnn_type", default="sage")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--gnn_layers", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_patches", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tcga_graph", default=f"{DATA}/Another_dataset/graphs_v2/feat_morph")
    ap.add_argument("--tcga_labels", default=f"{DATA}/labels.csv")
    ap.add_argument("--clinical", default=f"{DATA}/tcga_brca_complete.csv")
    ap.add_argument("--uch_graph", default=f"{DATA}/uch_graphs_v2/feat_morph")
    ap.add_argument("--uch_labels", default=f"{DATA}/uch_labels.csv")
    ap.add_argument("--out_dir", default=f"{RUNS}/full_tcga_model")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tcga_fm", default="",
                    help="TCGA patch-level foundation-model embedding cache (.pt)")
    ap.add_argument("--uch_fm", default="",
                    help="UCH cache, built with the SAME encoder/grid as --tcga_fm")
    ap.add_argument("--img_mode", default="concat",
                    choices=["concat", "only", "gated"])
    args = ap.parse_args()

    if bool(args.tcga_fm) != bool(args.uch_fm):
        raise SystemExit("--tcga_fm and --uch_fm must be given together: the UCH "
                         "model reuses the TCGA-trained image projection.")

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    # ── ALL TCGA (patient-grouped val slice only) ────────────────────────────
    full = SlideGraphDataset(args.tcga_graph, args.tcga_labels,
                             fm_emb=args.tcga_fm or None)
    in_dim = full.feature_dim()
    img_dim = full.img_dim()
    cont = load_continuous(args.clinical) if args.task == "regression" else {}
    by_pat = defaultdict(list)
    for i, (stem, _, _) in enumerate(full.items):
        by_pat[_patient_barcode(stem)].append(i)
    pats = sorted(by_pat); rng = np.random.RandomState(args.seed); rng.shuffle(pats)
    nval = max(1, int(len(pats) * args.val_frac)); vpats = set(pats[:nval])
    tr = [i for p in pats if p not in vpats for i in by_pat[p]]
    va = [i for p in pats if p in vpats for i in by_pat[p]]
    print(f"Train on ALL TCGA: {len(tr)} train / {len(va)} val slides "
          f"({len(full.items)} total, patient-grouped)")

    train_ds = copy.copy(full); train_ds.train = True; train_ds.max_patches = args.max_patches
    trl = DataLoader(Subset(train_ds, tr), batch_size=args.batch_size, shuffle=True,
                     collate_fn=collate_slides, num_workers=4)
    val = DataLoader(Subset(full, va), batch_size=args.batch_size, shuffle=False,
                     collate_fn=collate_slides, num_workers=4)

    model = RecurrenceModel(in_dim=in_dim, hidden=args.hidden, gnn_layers=args.gnn_layers,
                            gnn_type=args.gnn_type, aggregator=args.aggregator,
                            dropout=args.dropout, img_dim=img_dim,
                            img_mode=args.img_mode).to(device)
    if args.task == "classification":
        lab = [full.labels[i] for i in tr]; npos = max(1, sum(lab))
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(lab)-npos)/npos], device=device))
    else:
        crit = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.01)

    best_auc, best_state = -1, None
    for ep in range(1, args.epochs+1):
        run_epoch(model, trl, device, crit, args.task, cont, opt)
        _, va_auc = run_epoch(model, val, device, crit, args.task, cont)
        sched.step()
        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc, best_state = va_auc, copy.deepcopy(model.state_dict())
        if ep % 20 == 0:
            print(f"  ep {ep}: val AUROC {va_auc:.4f} (best {best_auc:.4f})")
    model.load_state_dict(best_state)
    torch.save({"model_state": best_state, "in_dim": in_dim, **vars(args)},
               os.path.join(args.out_dir, "full_tcga_model.pt"))
    print(f"Saved full-TCGA model (val AUROC {best_auc:.4f})")

    # ── infer on UCH ─────────────────────────────────────────────────────────
    uds = SlideGraphDataset(args.uch_graph, args.uch_labels,
                            fm_emb=args.uch_fm or None)
    if uds.img_dim() != img_dim:
        raise SystemExit(f"UCH image dim {uds.img_dim()} != TCGA {img_dim}; "
                         "the two caches must come from the same encoder.")
    print("UCH:", uds.summary())
    ul = DataLoader(uds, batch_size=4, shuffle=False, collate_fn=collate_slides, num_workers=4)
    model.eval(); scores, sids = [], []
    with torch.no_grad():
        for batch, p2s, y, sd in ul:
            out = model(batch.to(device), p2s.to(device), num_slides=y.numel())
            s = torch.sigmoid(out) if args.task == "classification" else out
            scores.extend(s.cpu().tolist()); sids.extend(sd)
    ss = dict(zip(sids, scores))

    lab = pd.read_csv(args.uch_labels); lab["slide_id"] = lab["slide_id"].astype(str)
    rows = [(r.slide_id, ss[r.slide_id], int(r.label), float(r.rs))
            for r in lab.itertuples() if r.slide_id in ss]
    sid, sc, y, rs = [list(z) for z in zip(*rows)]
    sc, y, rs = np.array(sc), np.array(y), np.array(rs)
    if _HAS and len(set(y)) > 1:
        auc = roc_auc_score(y, sc); sp = spearmanr(sc, rs)[0]
        fpr, tpr, t = roc_curve(y, sc); thr = t[np.argmax(tpr-fpr)]
        pred = (sc >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        print(f"\n=== UCH external validation (full-TCGA model) ===")
        print(f"  n {len(y)} (High {y.sum()} / Low {(y==0).sum()})")
        print(f"  AUROC        : {auc:.4f}   (3-fold ensemble was 0.646; paper 0.798)")
        print(f"  Spearman(RS) : {sp:.4f}")
        print(f"  accuracy {(tp+tn)/len(y):.4f}  recall {tp/(tp+fn):.4f}  "
              f"precision {tp/(tp+fp) if tp+fp else 0:.4f}  specificity {tn/(tn+fp):.4f}")
        print(f"  confusion: TN {tn} FP {fp} FN {fn} TP {tp}")
        with open(os.path.join(args.out_dir, "uch_validation.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["slide_id","score","label","rs","pred"])
            for a,b_,c,d_,e in zip(sid,sc,y,rs,pred): w.writerow([a,f"{b_:.4f}",int(c),d_,int(e)])
        print(f"  saved -> {args.out_dir}/uch_validation.csv")
        metrics = {
            "cohort": "UCH", "n": int(len(y)),
            "n_high": int(y.sum()), "n_low": int((y == 0).sum()),
            "auroc": float(auc), "spearman_rs": float(sp),
            "threshold_youden": float(thr),
            "accuracy": float((tp + tn) / len(y)),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else 0.0,
            "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
            "config": {"task": args.task, "aggregator": args.aggregator,
                       "gnn_type": args.gnn_type, "hidden": args.hidden, "lr": args.lr,
                       # img_mode is inert unless an embedding cache was given
                       "image_branch": args.img_mode if args.tcga_fm else "none",
                       "tcga_graph": args.tcga_graph, "uch_graph": args.uch_graph},
        }
        with open(os.path.join(args.out_dir, "uch_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  saved -> {args.out_dir}/uch_metrics.json")


if __name__ == "__main__":
    main()
