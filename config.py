from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from paths import DATA, RUNS


@dataclass
class Config:
    # ── data ────────────────────────────────────────────────────────────────
    graph_dir:  str = f"{DATA}/Another_dataset/graphs_v2/feat_morph"   # contains part*/  subdirs
    labels_csv: str = f"{DATA}/labels.csv"

    # ── model ───────────────────────────────────────────────────────────────
    in_dim:     Optional[int] = None   # auto-detected from first graph
    hidden:     int   = 256
    gnn_layers: int   = 3
    gnn_type:   str   = "gat"          # "gat" | "sage"
    aggregator: str   = "mean"         # "mean" | "attention" | "slide_gnn"
    dropout:    float = 0.25
    # 0 = use all patches; >0 = random subsample per slide during training
    max_patches_per_slide: int = 0

    # ── training ────────────────────────────────────────────────────────────
    epochs:       int   = 60
    batch_size:   int   = 4            # slides per minibatch
    lr:           float = 1e-3
    weight_decay: float = 1e-4
    val_frac:     float = 0.15
    test_frac:    float = 0.15
    seed:         int   = 42
    num_workers:  int   = 0
    device:       str   = "cuda"

    # ── i/o ─────────────────────────────────────────────────────────────────
    out_dir:   str = RUNS   # actual subdir = out_dir/aggregator_gnntype
    ckpt_name: str = "best.pt"
