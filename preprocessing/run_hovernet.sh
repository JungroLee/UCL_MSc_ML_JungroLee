#!/usr/bin/env bash
# Step 2/3 — nuclei detection + classification with HoVer-Net (PanNuke, fast mode).
#
# Input : a folder of super-resolved *.png tiles (output of super_res.py).
# Output: <out>/json/<tile>.json  — one nucleus record per instance, with
#         centroid, type, type_prob, contour, bbox. build_graphs.py reads these.
#
# HoVer-Net also writes <out>/mat/ (per-tile .mat) which we do NOT use and which
# is large; delete it after a run to save disk/inode quota:  rm -rf <out>/mat
#
# The HoVer-Net code + PanNuke checkpoint are bundled inside this package under
# ../external/, so this runs from a Dissertation_final checkout alone (they are
# git-ignored for size — restore with fetch_hovernet.sh if missing). Override
# with HOVERNET_DIR / HOVERNET_CKPT to point elsewhere.
#
#   bash run_hovernet.sh <sr_tiles_dir> <out_dir>
set -e

IN_DIR="${1:?usage: run_hovernet.sh <sr_tiles_dir> <out_dir>}"
OUT_DIR="${2:?usage: run_hovernet.sh <sr_tiles_dir> <out_dir>}"

# resolve ../external relative to THIS script, so cwd doesn't matter
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT="$HERE/../external"
HOVERNET="${HOVERNET_DIR:-$EXT/hover_net}"
CKPT="${HOVERNET_CKPT:-$EXT/pretrained/hovernet_fast_pannuke_type_tf2pytorch.tar}"

python3 "$HOVERNET/run_infer.py" \
  --gpu="0" \
  --nr_types=6 \
  --type_info_path="$HOVERNET/type_info.json" \
  --batch_size=16 \
  --model_mode=fast \
  --model_path="$CKPT" \
  --nr_inference_workers=4 \
  --nr_post_proc_workers=8 \
  tile \
  --input_dir="$IN_DIR" \
  --output_dir="$OUT_DIR"

echo "done -> $OUT_DIR/json/   (delete $OUT_DIR/mat to reclaim disk)"
