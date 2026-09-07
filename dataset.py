"""
SlideGraphDataset: groups per-patch cell graphs into per-SLIDE bags and attaches
a recurrence label.

On-disk layout expected (output of graph/batch_build_graph_embed.py):
    <graph_dir>/<slide_stem>__x{X}_y{Y}.pt      # one PyG Data per patch

Each .pt is a torch_geometric Data with:
    x          [N_cells, D]   node (cell) features
    edge_index [2, E]         spatial kNN edges
    pos        [N_cells, 2]   centroids
    y          [N_cells]      HoVer-Net cell type  (NOT used as the target here)

A "sample" returned by this dataset is ONE slide = a list of patch Data objects
plus a single slide-level binary recurrence label.

Labels CSV: columns `slide_id,label`. `slide_id` is matched against the full
slide stem first, then against the 12-char TCGA patient barcode
(e.g. TCGA-S3-AA15) so you can label at the patient level.
"""
import os
import glob
import csv
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch


# torch>=2.6 defaults torch.load(weights_only=True), which cannot unpickle a
# PyG Data object. Our graphs are produced locally and trusted, so load fully.
def _load_graph(path):
    return torch.load(path, weights_only=False)


def _slide_stem(filename):
    """`TCGA-..DX1.ABC__x0_y1024.pt` -> `TCGA-..DX1.ABC`."""
    base = os.path.splitext(os.path.basename(filename))[0]
    return base.split("__")[0]


def _patient_barcode(slide_stem):
    """First 12 chars of a TCGA id, e.g. `TCGA-S3-AA15`. Safe for non-TCGA ids."""
    return slide_stem[:12]


def read_labels(labels_csv):
    """Return dict mapping slide_id (as written in the CSV) -> int label."""
    mapping = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        if "slide_id" not in reader.fieldnames or "label" not in reader.fieldnames:
            raise ValueError(
                f"{labels_csv} must have columns 'slide_id,label'; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            sid = row["slide_id"].strip()
            lab = row["label"].strip()
            if sid == "" or lab == "":
                continue
            mapping[sid] = int(float(lab))
    return mapping


def discover_slides(graph_dir):
    """Group all patch .pt files by slide stem. Returns {slide_stem: [paths]}.
    Supports both flat layout (graph_dir/*.pt) and part-subdirectory layout
    (graph_dir/part*/*.pt) produced by batch_build_graph_multi.py.
    """
    paths = sorted(glob.glob(os.path.join(graph_dir, "*.pt")))
    if not paths:
        paths = sorted(glob.glob(os.path.join(graph_dir, "**", "*.pt"), recursive=True))
    slides = defaultdict(list)
    for p in paths:
        slides[_slide_stem(p)].append(p)
    return dict(slides)


def load_fm_embeddings(path):
    """Load the patch-level foundation-model cache written by
    `code/preprocessing/embed_patches_fm.py`. Returns (stem -> row index, [N, D] tensor)."""
    blob = torch.load(path, weights_only=False)
    index = {s: i for i, s in enumerate(blob["stems"])}
    return index, blob["emb"]


class SlideGraphDataset(Dataset):
    def __init__(self, graph_dir, labels_csv, max_patches_per_slide=0, train=False,
                 fm_emb=None):
        """fm_emb: optional path to a patch-level foundation-model embedding cache.
        When given, every patch graph gets a `.img` attribute of shape [1, D];
        PyG's Batch then stacks these into [num_patches, D] automatically, so the
        training loops need no changes."""
        super().__init__()
        self.max_patches = max_patches_per_slide
        self.train = train

        self.fm_index, self.fm_emb = (None, None)
        if fm_emb:
            self.fm_index, self.fm_emb = load_fm_embeddings(fm_emb)

        label_map = read_labels(labels_csv)
        slides = discover_slides(graph_dir)

        self.items = []          # list of (slide_stem, [patch_paths], label)
        n_unlabelled = 0
        for stem, paths in slides.items():
            if stem in label_map:
                lab = label_map[stem]
            elif _patient_barcode(stem) in label_map:
                lab = label_map[_patient_barcode(stem)]
            else:
                n_unlabelled += 1
                continue
            self.items.append((stem, paths, lab))

        if not self.items:
            raise RuntimeError(
                f"No slides in {graph_dir} matched a label in {labels_csv}. "
                "Check that slide_id values line up (full stem or 12-char barcode)."
            )
        self._unlabelled = n_unlabelled

    # -- introspection helpers ------------------------------------------
    @property
    def labels(self):
        return [lab for _, _, lab in self.items]

    def feature_dim(self):
        """Peek at the first patch graph to get node feature dim D."""
        _, paths, _ = self.items[0]
        return _load_graph(paths[0]).x.shape[1]

    def img_dim(self):
        """Dim of the attached patch-level image embedding (0 if none)."""
        return 0 if self.fm_emb is None else int(self.fm_emb.shape[1])

    def summary(self):
        pos = sum(self.labels)
        return (f"{len(self.items)} slides "
                f"({pos} recurred / {len(self.items) - pos} not), "
                f"{self._unlabelled} slides skipped (no label)")

    # -- Dataset API -----------------------------------------------------
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        stem, paths, lab = self.items[idx]
        if self.train and self.max_patches and len(paths) > self.max_patches:
            sel = torch.randperm(len(paths))[: self.max_patches].tolist()
            paths = [paths[i] for i in sel]
        patches = [_load_graph(p) for p in paths]
        if self.fm_emb is not None:
            for p, g in zip(paths, patches):
                pstem = os.path.splitext(os.path.basename(p))[0]
                row = self.fm_index.get(pstem)
                g.img = (self.fm_emb[row].float().unsqueeze(0) if row is not None
                         else torch.zeros(1, self.fm_emb.shape[1]))
        return patches, lab, stem


def collate_slides(samples):
    """
    Flatten a minibatch of slides into:
        batch         : a single PyG Batch of ALL patches in the minibatch.
                        batch.batch maps each NODE -> its patch index (0..P-1).
        patch_to_slide: LongTensor[P] mapping each PATCH -> its slide index (0..S-1).
        y             : FloatTensor[S] binary labels.
        slide_ids     : list[str] of length S.
    """
    all_patches, patch_to_slide, ys, slide_ids = [], [], [], []
    for s_idx, (patches, lab, stem) in enumerate(samples):
        all_patches.extend(patches)
        patch_to_slide.extend([s_idx] * len(patches))
        ys.append(float(lab))
        slide_ids.append(stem)

    batch = Batch.from_data_list(all_patches)
    return (
        batch,
        torch.tensor(patch_to_slide, dtype=torch.long),
        torch.tensor(ys, dtype=torch.float),
        slide_ids,
    )
