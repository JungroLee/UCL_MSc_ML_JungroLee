"""
Recurrence prediction: three aggregation strategies sharing one GNN backbone.

    Method 1  mean       MeanPoolAggregator    mean+max pool, no learned params
    Method 2  attention  AttentionMILAggregator gated ABMIL (Ilse et al. 2018)
    Method 3  slide_gnn  SlideGNNAggregator     slide-level kNN GNN (Patch-GCN style)

Shared:
    PatchGNN: GNN applied to each cell graph -> per-patch embedding
              supports gnn_type = "gat" | "sage"

Select via RecurrenceModel(aggregator="mean"|"attention"|"slide_gnn").
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, GINConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import softmax as pyg_softmax


# ─────────────────────────────────────────────────────────────────────────────
# Shared patch-level GNN encoder
# ─────────────────────────────────────────────────────────────────────────────

class PatchGNN(nn.Module):
    """
    GNN backbone applied independently to every patch cell-graph.
    Readout: global mean‖max concat → projection → patch embedding [hidden].

    gnn_type = "gat"  : Graph Attention Network (4 heads)
    gnn_type = "sage" : GraphSAGE
    gnn_type = "gcn"  : Graph Convolutional Network
    gnn_type = "gin"  : Graph Isomorphism Network (2-layer MLP per conv)
    """

    def __init__(self, in_dim: int, hidden: int, n_layers: int = 3,
                 gnn_type: str = "gat", dropout: float = 0.25):
        super().__init__()
        if gnn_type not in ("gat", "sage", "gcn", "gin"):
            raise ValueError(f"gnn_type must be gat|sage|gcn|gin, got '{gnn_type}'")
        self.dropout  = dropout
        self.gnn_type = gnn_type
        self.convs    = nn.ModuleList()
        self.norms    = nn.ModuleList()

        d = in_dim
        for _ in range(n_layers):
            if gnn_type == "gat":
                heads = 4
                self.convs.append(
                    GATConv(d, hidden // heads, heads=heads,
                            dropout=dropout, concat=True)
                )
            elif gnn_type == "sage":
                self.convs.append(SAGEConv(d, hidden))
            elif gnn_type == "gcn":
                self.convs.append(GCNConv(d, hidden))
            else:  # gin
                mlp = nn.Sequential(
                    nn.Linear(d, hidden), nn.ReLU(),
                    nn.Linear(hidden, hidden),
                )
                self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.BatchNorm1d(hidden))
            d = hidden

        self.proj = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, x, edge_index, node_to_patch):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = torch.cat([
            global_mean_pool(x, node_to_patch),
            global_max_pool(x, node_to_patch),
        ], dim=1)                           # [num_patches, 2*hidden]
        return self.proj(h)                 # [num_patches, hidden]


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Mean‖Max Pooling  (simple baseline)
# ─────────────────────────────────────────────────────────────────────────────

class MeanPoolAggregator(nn.Module):
    """
    Treats every patch equally.
    Concatenates global mean and max of patch embeddings → slide embedding.
    No learnable parameters.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.out_dim = 2 * hidden

    def forward(self, patch_emb, patch_to_slide, num_slides, **_):
        return torch.cat([
            global_mean_pool(patch_emb, patch_to_slide, size=num_slides),
            global_max_pool(patch_emb, patch_to_slide, size=num_slides),
        ], dim=1)                           # [num_slides, 2*hidden]


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Gated Attention MIL  (Ilse et al. 2018 – ABMIL)
# ─────────────────────────────────────────────────────────────────────────────

class AttentionMILAggregator(nn.Module):
    """
    Gated attention MIL:
        a_i = softmax_j( V · (tanh(W·h_i) ⊙ sigmoid(U·h_i)) )
        slide = Σ_i  a_i · h_i

    Attention weights are interpretable — high-weight patches drive the
    slide-level prediction. Use get_attention_weights() to inspect them.
    """

    def __init__(self, hidden: int, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.out_dim  = hidden
        self.dropout  = dropout
        self.W = nn.Linear(hidden, attn_dim, bias=False)
        self.U = nn.Linear(hidden, attn_dim, bias=False)
        self.V = nn.Linear(attn_dim,      1, bias=False)

    def _logits(self, patch_emb):
        return self.V(
            torch.tanh(self.W(patch_emb)) * torch.sigmoid(self.U(patch_emb))
        ).squeeze(-1)                       # [num_patches]

    def forward(self, patch_emb, patch_to_slide, num_slides, **_):
        logits = self._logits(patch_emb)
        logits = F.dropout(logits, p=self.dropout, training=self.training)
        # per-slide softmax using torch_geometric scatter softmax
        a = pyg_softmax(logits, patch_to_slide, num_nodes=num_slides)  # [num_patches]

        slide = torch.zeros(num_slides, patch_emb.size(1), device=patch_emb.device)
        slide.scatter_add_(
            0,
            patch_to_slide.unsqueeze(1).expand_as(patch_emb),
            a.unsqueeze(1) * patch_emb,
        )
        return slide                        # [num_slides, hidden]

    @torch.no_grad()
    def get_attention_weights(self, patch_emb, patch_to_slide, num_slides):
        """Return per-patch attention weights for interpretability."""
        return pyg_softmax(self._logits(patch_emb), patch_to_slide,
                           num_nodes=num_slides)


# ─────────────────────────────────────────────────────────────────────────────
# Method 3: Slide-level GNN  (Patch-GCN, Chen et al. 2021)
# ─────────────────────────────────────────────────────────────────────────────

class SlideGNNAggregator(nn.Module):
    """
    Treats patch embeddings as nodes in a slide-level graph.
    Builds a kNN graph within each slide (in embedding space, or spatial if
    patch_pos is passed), then runs a lightweight GNN.
    Global mean pool → slide embedding.
    """

    def __init__(self, hidden: int, n_layers: int = 2, k: int = 8,
                 dropout: float = 0.25):
        super().__init__()
        self.k       = k
        self.out_dim = hidden
        self.convs   = nn.ModuleList([SAGEConv(hidden, hidden) for _ in range(n_layers)])
        self.norms   = nn.ModuleList([nn.BatchNorm1d(hidden) for _ in range(n_layers)])
        self.dropout = dropout

    def _build_edges(self, feat, batch_index):
        # patch count per slide is small (~32), so manual cdist is fine
        device = feat.device
        num_slides = int(batch_index.max().item()) + 1
        srcs, dsts = [], []
        for sid in range(num_slides):
            idx = (batch_index == sid).nonzero(as_tuple=True)[0]
            n = idx.numel()
            if n <= 1:
                continue
            k = min(self.k, n - 1)
            d = torch.cdist(feat[idx], feat[idx])   # [n, n]
            d.fill_diagonal_(float("inf"))
            _, nn_idx = d.topk(k, dim=1, largest=False)  # [n, k]
            src = idx.unsqueeze(1).expand(-1, k).reshape(-1)
            dst = idx[nn_idx.reshape(-1)]
            srcs.append(src); dsts.append(dst)
        if not srcs:
            return torch.zeros(2, 0, dtype=torch.long, device=device)
        return torch.stack([torch.cat(srcs), torch.cat(dsts)])

    def forward(self, patch_emb, patch_to_slide, num_slides,
                patch_pos=None, **_):
        # use spatial coords if available, else embedding space for kNN
        feat       = patch_pos if patch_pos is not None else patch_emb
        edge_index = self._build_edges(feat.detach(), patch_to_slide)

        x = patch_emb
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return global_mean_pool(x, patch_to_slide, size=num_slides)  # [num_slides, hidden]


# ─────────────────────────────────────────────────────────────────────────────
# Full model: PatchGNN + Aggregator + FC head
# ─────────────────────────────────────────────────────────────────────────────

_AGGREGATORS = {
    "mean":      MeanPoolAggregator,
    "attention": AttentionMILAggregator,
    "slide_gnn": SlideGNNAggregator,
}


class RecurrenceModel(nn.Module):
    """
    End-to-end pipeline:
        cell graph  →  PatchGNN  →  patch embedding
        patch embs  →  Aggregator →  slide embedding
        slide emb   →  FC head   →  recurrence logit
    """

    def __init__(self, in_dim: int, hidden: int = 256, gnn_layers: int = 3,
                 gnn_type: str = "gat", aggregator: str = "attention",
                 dropout: float = 0.25, img_dim: int = 0, img_mode: str = "concat"):
        """img_dim > 0 fuses a frozen patch-level foundation-model embedding
        (`batch.img`, see dataset.SlideGraphDataset(fm_emb=...)) with the cell-graph
        patch embedding, before the MIL aggregator:
            concat : project the image vector and mix it with the GNN embedding
            only   : ignore the graph and use the image embedding (ablation)
        """
        super().__init__()
        if aggregator not in _AGGREGATORS:
            raise ValueError(f"aggregator must be one of {list(_AGGREGATORS)}, got '{aggregator}'")
        if img_mode not in ("concat", "only", "gated"):
            raise ValueError(f"img_mode must be concat|only|gated, got '{img_mode}'")

        self.gnn = PatchGNN(in_dim, hidden, gnn_layers, gnn_type, dropout)
        self.img_dim, self.img_mode = img_dim, img_mode
        if img_dim > 0:
            self.img_proj = nn.Sequential(
                nn.LayerNorm(img_dim), nn.Linear(img_dim, hidden), nn.ReLU(),
                nn.Dropout(dropout),
            )
            if img_mode == "concat":
                self.fuse = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
            elif img_mode == "gated":
                # additive fusion with a single learned scalar on the image branch:
                #     patch_emb = img_gate * img + LayerNorm(graph)
                # the head LayerNorms the pooled vector, so absolute scale washes
                # out and img_gate reads directly as the image:graph weight ratio
                # (>1 image-dominated, <1 graph-dominated). Both branches are
                # normalised so neither starts with a scale advantage.
                self.graph_norm = nn.LayerNorm(hidden)
                self.img_gate = nn.Parameter(torch.ones(1))
        self.agg = _AGGREGATORS[aggregator](hidden)

        self.head = nn.Sequential(
            nn.LayerNorm(self.agg.out_dim),
            nn.Linear(self.agg.out_dim, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch, patch_to_slide, num_slides, **kwargs):
        patch_emb = self.gnn(batch.x, batch.edge_index, batch.batch)
        if self.img_dim > 0:
            img = self.img_proj(batch.img.float())            # [num_patches, hidden]
            if self.img_mode == "only":
                patch_emb = img
            elif self.img_mode == "gated":
                patch_emb = self.img_gate * img + self.graph_norm(patch_emb)
            else:
                patch_emb = self.fuse(torch.cat([patch_emb, img], dim=1))
        slide_emb = self.agg(patch_emb, patch_to_slide, num_slides, **kwargs)
        return self.head(slide_emb).squeeze(-1)   # [num_slides]
