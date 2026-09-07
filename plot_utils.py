"""
Plotting helpers (matplotlib only, no seaborn).

    plot_training_curves(history_csv, out_png)  -> loss/acc/AUC vs epoch
    plot_confusion_matrix(cm, out_png, title)   -> 2x2 confusion heatmap
    plot_sweep_comparison(results_csv, out_png) -> grouped bar chart of test AUC
"""
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_curves(history_csv, out_png, title=""):
    ep, trl, tra, tru, val, vaa, vau = [], [], [], [], [], [], []
    with open(history_csv) as f:
        for r in csv.DictReader(f):
            ep.append(int(r["epoch"]))
            trl.append(float(r["tr_loss"])); tra.append(float(r["tr_acc"])); tru.append(float(r["tr_auc"]))
            val.append(float(r["va_loss"])); vaa.append(float(r["va_acc"])); vau.append(float(r["va_auc"]))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (name, tr, va) in zip(axes, [
        ("Loss", trl, val), ("Accuracy", tra, vaa), ("AUC", tru, vau)
    ]):
        ax.plot(ep, tr, label="train", color="#4C9BE8", linewidth=1.8)
        ax.plot(ep, va, label="val",   color="#E85C5C", linewidth=1.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel(name); ax.set_title(name)
        ax.grid(alpha=0.25); ax.legend()
        if name == "AUC":
            best_ep = ep[int(np.argmax(va))]
            best_va = max(va)
            ax.axvline(best_ep, color="green", ls="--", alpha=0.6)
            ax.annotate(f"best val AUC {best_va:.3f}\n@ep {best_ep}",
                        xy=(best_ep, best_va), fontsize=9, color="green")

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm, out_png, title="Confusion Matrix",
                          class_names=("Low (0)", "High (1)")):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_sweep_comparison(results_csv, out_png):
    """Grouped bar chart: best test AUC per (gnn_type x aggregator)."""
    rows = []
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return

    gnn_types   = sorted({r["gnn_type"] for r in rows})
    aggregators = sorted({r["aggregator"] for r in rows})

    # best test AUC per (gnn, agg) across HP
    best = {}
    for r in rows:
        key = (r["gnn_type"], r["aggregator"])
        auc = float(r["test_auc"])
        if key not in best or auc > best[key]:
            best[key] = auc

    x = np.arange(len(gnn_types))
    w = 0.8 / len(aggregators)
    colors = ["#4C9BE8", "#E85C5C", "#6BBF59", "#E8A64C"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for k, agg in enumerate(aggregators):
        vals = [best.get((g, agg), 0.0) for g in gnn_types]
        bars = ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=agg,
                      color=colors[k % len(colors)], alpha=0.85)
        for b, v in zip(bars, vals):
            if v > 0:
                ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x); ax.set_xticklabels([g.upper() for g in gnn_types])
    ax.set_ylabel("Best Test AUC")
    ax.set_title("Best Test AUC per Graph Backbone × Aggregator")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="grey", ls=":", alpha=0.6)
    ax.legend(title="Aggregator")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
