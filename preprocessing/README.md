# Preprocessing

Turns raw pathology tiles (png) into the cell graphs (.pt) the GNN sweep uses. Three steps:

1. **`super_res.py`** — 4x super-resolves tiles with Swin2SR (so HoVer-Net can find nuclei reliably).
2. **`run_hovernet.sh`** — runs HoVer-Net to detect/classify nuclei → saves JSON.
3. **`build_graphs.py`** — builds a cell graph (nuclei as nodes) from the JSON, producing all three edge variants at once: knn6 / delaunay / radius110.

## Run

On your own tiles, fill in the paths:

```bash
cd Dissertation_final/preprocessing

python3 super_res.py --in <tiles_dir> --out <sr_tiles_dir>

bash run_hovernet.sh <sr_tiles_dir> <hovernet_out_dir>

python3 build_graphs.py \
    --json_dir <hovernet_out_dir>/json \
    --part part00 \
    --base_out <cohort>/graphs_v2
```

`--base_out` is where the resulting graphs get saved, and it's the same path that `../paths.py` reads from.

## Try it now — copy/paste demo

No setup, no GPU needed for step 3. `../sample_data/preprocessing_demo/` already
has 8 TCGA tiles at every stage (post-SR tiles, HoVer-Net JSON, and built
graphs), so this runs as-is:

```bash
cd Dissertation_final/preprocessing

# step 2: HoVer-Net on the demo tiles (needs a GPU)
bash run_hovernet.sh ../sample_data/preprocessing_demo/tcga/tiles /tmp/demo_hovernet_out

# step 3: build graphs from that JSON (no GPU needed)
python3 build_graphs.py \
    --json_dir /tmp/demo_hovernet_out/json \
    --part part00 \
    --base_out /tmp/demo_graphs_out
```

Or skip step 2 (it needs a GPU) and just re-run step 3 on the JSON already
bundled in the demo:

```bash
python3 build_graphs.py \
    --json_dir ../sample_data/preprocessing_demo/tcga/hovernet_json \
    --part part00 \
    --base_out /tmp/demo_graphs_out
```

Swap `tcga` for `uch` to try the UCH demo tiles instead. See
`../sample_data/preprocessing_demo/{tcga,uch}/{tiles,hovernet_json,graphs}/`
for what each stage's output looks like.
