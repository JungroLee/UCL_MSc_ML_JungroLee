import random
from collections import defaultdict

from dataset import _patient_barcode
from paths import DATA

try:
    from sklearn.model_selection import StratifiedGroupKFold
    _HAS_SGK = True
except ImportError:
    _HAS_SGK = False


def _patient_groups(items):
    """Return (groups, patient_label) where groups[i] is item i's patient barcode
    and patient_label[pid] is that patient's label (first slide's label; a patient's
    slides share a label since labels are assigned at patient level)."""
    groups = [_patient_barcode(stem) for stem, _, _ in items]
    patient_label = {}
    for (stem, _, lab), pid in zip(items, groups):
        patient_label.setdefault(pid, lab)
    return groups, patient_label


def grouped_stratified_split(items, val_frac=0.15, test_frac=0.15, seed=42):
    """Patient-grouped, label-stratified split. Returns (tr_idx, va_idx, te_idx)
    as lists of ITEM indices into `items`."""
    groups, patient_label = _patient_groups(items)
    patients = sorted(patient_label.keys())

    if _HAS_SGK:
        idx = _sgk_split(items, groups, val_frac, test_frac, seed)
    else:
        idx = _manual_split(items, groups, patient_label, patients,
                            val_frac, test_frac, seed)
    assert_no_leakage(items, *idx)
    return idx


def _sgk_split(items, groups, val_frac, test_frac, seed):
    """Use StratifiedGroupKFold twice: carve test first, then val from the rest."""
    import numpy as np
    y = np.array([lab for _, _, lab in items])
    g = np.array(groups)
    all_idx = np.arange(len(items))

    # ---- test fold ----
    n_test_folds = max(2, round(1.0 / max(test_frac, 1e-6)))
    sgk = StratifiedGroupKFold(n_splits=n_test_folds, shuffle=True, random_state=seed)
    trainval_i, test_i = next(sgk.split(all_idx, y, g))

    # ---- val fold (from train+val portion) ----
    y2, g2 = y[trainval_i], g[trainval_i]
    rel_val = val_frac / (1.0 - test_frac)
    n_val_folds = max(2, round(1.0 / max(rel_val, 1e-6)))
    sgk2 = StratifiedGroupKFold(n_splits=n_val_folds, shuffle=True, random_state=seed)
    tr_rel, va_rel = next(sgk2.split(trainval_i, y2, g2))

    tr = trainval_i[tr_rel].tolist()
    va = trainval_i[va_rel].tolist()
    te = test_i.tolist()
    return tr, va, te


def _manual_split(items, groups, patient_label, patients, val_frac, test_frac, seed):
    """Fallback: shuffle patients within each label, allocate by fraction."""
    by_label = defaultdict(list)
    for pid in patients:
        by_label[patient_label[pid]].append(pid)

    rng = random.Random(seed)
    tr_pat, va_pat, te_pat = set(), set(), set()
    for lab, plist in by_label.items():
        plist = plist[:]
        rng.shuffle(plist)
        n = len(plist)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        te_pat.update(plist[:n_test])
        va_pat.update(plist[n_test:n_test + n_val])
        tr_pat.update(plist[n_test + n_val:])

    tr, va, te = [], [], []
    for i, pid in enumerate(groups):
        if pid in te_pat:
            te.append(i)
        elif pid in va_pat:
            va.append(i)
        else:
            tr.append(i)
    return tr, va, te


def assert_no_leakage(items, tr, va, te):
    """Raise if any patient barcode appears in more than one split."""
    groups = [_patient_barcode(stem) for stem, _, _ in items]
    s_tr = {groups[i] for i in tr}
    s_va = {groups[i] for i in va}
    s_te = {groups[i] for i in te}
    overlap = (s_tr & s_va) | (s_tr & s_te) | (s_va & s_te)
    if overlap:
        raise AssertionError(
            f"Patient leakage across splits: {sorted(overlap)[:10]} "
            f"({len(overlap)} patients in >1 split)"
        )


def split_summary(items, tr, va, te):
    groups = [_patient_barcode(stem) for stem, _, _ in items]
    labels = [lab for _, _, lab in items]

    def stat(idx):
        pats = {groups[i] for i in idx}
        pos = sum(labels[i] for i in idx)
        return len(idx), len(pats), pos, len(idx) - pos

    lines = [f"{'split':<6} {'slides':>7} {'patients':>9} {'pos':>5} {'neg':>5}"]
    lines.append("-" * 36)
    for name, idx in [("train", tr), ("val", va), ("test", te)]:
        s, p, pos, neg = stat(idx)
        lines.append(f"{name:<6} {s:>7} {p:>9} {pos:>5} {neg:>5}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from dataset import SlideGraphDataset

    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_dir",  default=f"{DATA}/Another_dataset/graphs_v2/feat_morph")
    ap.add_argument("--labels_csv", default=f"{DATA}/labels.csv")
    ap.add_argument("--val_frac",   type=float, default=0.15)
    ap.add_argument("--test_frac",  type=float, default=0.15)
    ap.add_argument("--seed",       type=int,   default=42)
    args = ap.parse_args()

    ds = SlideGraphDataset(args.graph_dir, args.labels_csv)
    print("Dataset:", ds.summary())
    print(f"StratifiedGroupKFold available: {_HAS_SGK}\n")

    tr, va, te = grouped_stratified_split(ds.items, args.val_frac,
                                          args.test_frac, args.seed)
    print(split_summary(ds.items, tr, va, te))
    print("\nLeakage guard passed: no patient appears in more than one split.")
