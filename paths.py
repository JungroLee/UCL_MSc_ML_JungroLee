import os

HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DATA = os.path.join(HERE, "sample_data")

DATA = os.environ.get("DISS_DATA", _DEFAULT_DATA)
OUT  = os.environ.get("DISS_OUT",
                      os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"))
RUNS = os.path.join(OUT, "runs")
LOGS = os.path.join(OUT, "logs")

# ── cohorts ────────────────────────────────────────────────────────────────
# TCGA-BRCA training cohorts: the same slides, different patch bags.
COHORTS = {
    "random32":    os.path.join(DATA, "Another_dataset"),     # 32 random patches / slide
    "annotated32": os.path.join(DATA, "Annotated_Dataset"),   # 32 pathologist-annotated
}

# ── edge-construction methods (node features are identical 13-D morphology) ──
EDGES = {
    "knn6":      "feat_morph",       # k-nearest-neighbour, k=6
    "delaunay":  "feat_delaunay",    # Delaunay triangulation
    "radius110": "feat_radius110",   # fixed 110 px radius
}

AGGREGATORS = ["mean", "attention", "slide_gnn"]
BACKBONES   = ["gcn", "sage", "gat", "gin"]

# ── labels (shared by both TCGA cohorts) ───────────────────────────────────
LABELS   = os.path.join(DATA, "labels.csv")
CLINICAL = os.path.join(DATA, "tcga_brca_complete.csv")

# ── UCH external cohort ────────────────────────────────────────────────────
UCH_GRAPHS = os.path.join(DATA, "uch_graphs_v2")
UCH_LABELS = os.path.join(DATA, "uch_labels.csv")
UCH_FM     = os.path.join(DATA, "uch_fm_emb", "uni2h_grid4.pt")


def graph_dir(cohort, edge):
    """Cell graphs for one cohort under one edge-construction method."""
    return os.path.join(COHORTS[cohort], "graphs_v2", EDGES[edge])


def fm_emb(cohort):
    """UNI2-h patch embeddings for the image branch of one cohort."""
    return os.path.join(COHORTS[cohort], "fm_emb", "uni2h_grid4.pt")


def uch_graph_dir(edge):
    return os.path.join(UCH_GRAPHS, EDGES[edge])
