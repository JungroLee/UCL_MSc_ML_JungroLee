"""
Step 3/3 — HoVer-Net JSON -> cell graphs, one per tile.

Nodes are nuclei, node features are the 13-D morphology vector the models use:

    [ norm_x, norm_y,            # centroid, z-normalised per tile
      type_onehot x 6,           # HoVer-Net PanNuke class
      type_prob,                 # classifier confidence
      norm_area, norm_perimeter, # contour, log1p then z-normalised per tile
      aspect_ratio, convexity ]  # bbox h/w (clamped); area / convex-hull area

The three edge constructions the sweep compares are built here from the SAME
nuclei / features, differing only in `edge_index`:

    knn6       feat_morph/       k=6 nearest neighbours (density-blind)
    delaunay   feat_delaunay/    parameter-free triangulation (follows tissue)
    radius110  feat_radius110/   all pairs within 110 px (exposes real density;
                                 valid across cohorts because both are x4-SR)

Output layout matches paths.py exactly:

    <base_out>/feat_morph/<part>/<tile>.pt
    <base_out>/feat_delaunay/<part>/<tile>.pt
    <base_out>/feat_radius110/<part>/<tile>.pt

Resume-safe: a tile whose .pt already exists for every requested edge is skipped.

    python3 build_graphs.py --json_dir <hovernet_out>/json --part part00 \
        --base_out <cohort>/graphs_v2
    python3 build_graphs.py --json_dir ... --part ... --base_out ... \
        --edges knn6 delaunay          # subset; default is all three
"""
import argparse, json, os, glob
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree, ConvexHull, Delaunay
import torch
from torch_geometric.data import Data

NUM_TYPES = 6
# edge name -> output subfolder (kept identical to Dissertation_final/paths.py:EDGES)
EDGE_DIRS = {"knn6": "feat_morph", "delaunay": "feat_delaunay", "radius110": "feat_radius110"}
RADIUS_PX = 110.0
KNN_K = 6


# ── geometry helpers ─────────────────────────────────────────────────────────

def _area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _perimeter(pts):
    return float(np.sum(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1)))


def _convexity(pts, area):
    if len(pts) < 3:
        return 1.0
    try:
        return float(np.clip(area / (ConvexHull(pts).volume + 1e-6), 0.0, 1.0))
    except Exception:
        return 1.0


def _aspect_ratio(bbox):
    (r0, c0), (r1, c1) = bbox[0], bbox[1]
    return float(np.clip((abs(r1 - r0) + 1e-6) / (abs(c1 - c0) + 1e-6), 0.1, 10.0))


# ── nuclei loader ────────────────────────────────────────────────────────────

def load_nuclei(json_path):
    with open(json_path) as f:
        data = json.load(f)
    nuc = data["nuc"] if "nuc" in data else data

    coords, types, probs = [], [], []
    areas, perims, aspects, convex = [], [], [], []
    for info in nuc.values():
        coords.append(info["centroid"])
        types.append(int(info.get("type", 0)))
        probs.append(float(info.get("type_prob", 1.0)))
        pts = np.asarray(info.get("contour", []), dtype=np.float64)
        if len(pts) >= 3:
            a = _area(pts); p = _perimeter(pts); c = _convexity(pts, a)
        else:
            a, p, c = 0.0, 0.0, 1.0
        areas.append(a); perims.append(p); convex.append(c)
        aspects.append(_aspect_ratio(info.get("bbox", [[0, 0], [1, 1]])))

    coords = np.asarray(coords, dtype=np.float64)
    types = np.asarray(types, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    morph = np.stack([areas, perims, aspects, convex], axis=1)   # (N,4)
    return coords, types, probs, morph


# ── 13-D feat_morph node features ────────────────────────────────────────────

def _znorm(a):
    return (a - a.mean(0)) / (a.std(0) + 1e-6)


def build_feat_morph(coords, types, probs, morph_raw):
    coords_n = _znorm(coords)                                    # (N,2)
    oh = np.zeros((len(types), NUM_TYPES))
    oh[np.arange(len(types)), np.clip(types, 0, NUM_TYPES - 1)] = 1.0
    morph = morph_raw.copy()
    morph[:, 0] = np.log1p(morph[:, 0])                          # area
    morph[:, 1] = np.log1p(morph[:, 1])                          # perimeter
    return np.concatenate([coords_n, oh, probs[:, None], _znorm(morph)], axis=1)  # (N,13)


# ── edge constructions ───────────────────────────────────────────────────────

def _knn(pos, k=KNN_K):
    if len(pos) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    _, idx = cKDTree(pos).query(pos, k=min(k + 1, len(pos)))
    src, dst = [], []
    for i, neigh in enumerate(np.atleast_2d(idx)):
        for j in np.atleast_1d(neigh)[1:]:
            src += [i, int(j)]; dst += [int(j), i]
    return np.asarray([src, dst], dtype=np.int64)


def _delaunay(pos):
    if len(pos) < 4:
        return _knn(pos)
    try:
        tri = Delaunay(pos)
    except Exception:                        # collinear / duplicate points
        return _knn(pos)
    e = set()
    for s in tri.simplices:
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = int(s[a]), int(s[b])
                e.add((i, j) if i < j else (j, i))
    if not e:
        return _knn(pos)
    src = [i for i, j in e] + [j for i, j in e]
    dst = [j for i, j in e] + [i for i, j in e]
    return np.asarray([src, dst], dtype=np.int64)


def _radius(pos, r=RADIUS_PX):
    # isolated nuclei keep degree 0 by design -- that sparsity IS the signal.
    if len(pos) < 2:
        return np.zeros((2, 0), dtype=np.int64)
    pairs = cKDTree(pos).query_pairs(r, output_type="ndarray")
    if len(pairs) == 0:
        return np.zeros((2, 0), dtype=np.int64)
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return np.asarray([src, dst], dtype=np.int64)


EDGE_FN = {"knn6": _knn, "delaunay": _delaunay, "radius110": _radius}


# ── per-tile builder ─────────────────────────────────────────────────────────

def build_one(json_path, out_dirs, edges):
    stem = Path(json_path).stem
    out_paths = {e: out_dirs[e] / f"{stem}.pt" for e in edges}
    if all(p.exists() for p in out_paths.values()):
        return "skipped"

    coords, types, probs, morph_raw = load_nuclei(json_path)
    if len(coords) < 2:
        return "empty"

    x = torch.tensor(build_feat_morph(coords, types, probs, morph_raw), dtype=torch.float)
    pos = torch.tensor(coords, dtype=torch.float)
    y = torch.tensor(types, dtype=torch.long)
    for e in edges:
        if out_paths[e].exists():
            continue
        ei = torch.tensor(np.ascontiguousarray(EDGE_FN[e](coords)), dtype=torch.long)
        torch.save(Data(x=x, edge_index=ei, pos=pos, y=y), out_paths[e])
    return "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json_dir", required=True, help="folder of HoVer-Net *.json")
    ap.add_argument("--part", required=True, help="part name, e.g. part00 (a subfolder per part)")
    ap.add_argument("--base_out", required=True, help="cohort's graphs_v2 root")
    ap.add_argument("--edges", nargs="+", default=list(EDGE_DIRS), choices=list(EDGE_DIRS))
    args = ap.parse_args()

    out_dirs = {}
    for e in args.edges:
        d = Path(args.base_out) / EDGE_DIRS[e] / args.part
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[e] = d

    jsons = sorted(glob.glob(os.path.join(args.json_dir, "*.json")))
    print(f"[{args.part}] {len(jsons)} JSON in {args.json_dir}  edges={args.edges}", flush=True)

    done = skipped = empty = failed = 0
    for i, jp in enumerate(jsons, 1):
        try:
            st = build_one(jp, out_dirs, args.edges)
        except Exception as ex:
            failed += 1
            print(f"  [{i}/{len(jsons)}] {Path(jp).stem}  FAILED: {ex}")
            continue
        done += st == "done"; skipped += st == "skipped"; empty += st == "empty"
        if done and done % 500 == 0:
            print(f"  [{i}/{len(jsons)}] {done} built...", flush=True)

    print(f"\n[{args.part}] done={done} skipped={skipped} empty={empty} failed={failed}")
    for e in args.edges:
        print(f"  -> {out_dirs[e]}/")


if __name__ == "__main__":
    main()
