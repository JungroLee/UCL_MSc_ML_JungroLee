# Cell-graph GNN sweep: edges × aggregation × backbone

Self-contained package that trains a cell-graph GNN on TCGA-BRCA and sweeps
every combination of:

| axis | values |
|---|---|
| edge construction | `knn6`, `delaunay`, `radius110` |
| aggregation | `mean`, `attention` (ABMIL), `slide_gnn` |
| GNN backbone | `gcn`, `sage`, `gat`, `gin` |

Each combination is scored twice: **TCGA 3-fold CV** (paper-matched, pooled
AUROC on held-out HR+/HER2⁻ patients) and **UCH external validation** (train
on all of TCGA, infer on UCH).

## Files

| file | role |
|---|---|
| `preprocessing/` | tiles → cell graphs (see its own README) |
| `paths.py` | data locations + the three grids |
| `run_sweep.py` | builds and runs the job list, resume-safe |
| `summarize.py` | collects every result into one table |
| `paper_eval.py` | TCGA 3-fold CV evaluation |
| `external_uch.py` | train on all TCGA, validate on UCH |
| `model.py` | GNN backbone + aggregator (+ optional image fusion) |
| `dataset.py` | per-patch graphs → per-slide bags + labels |
| `splits.py` | patient-grouped, label-stratified splits |
| `hrher2.py` | the paper's HR+/HER2⁻ patient filter |
| `train.py`, `config.py`, `plot_utils.py` | training loop, defaults, plots |
| `sample_data/` | small bundled dataset so the pipeline runs out of the box |

## Run it

No setup needed — just run it from inside this folder. It uses the bundled
`./sample_data/` by default:

```bash
python3 run_sweep.py --cohort random32 --dry_run   # list the jobs first
python3 run_sweep.py --cohort random32             # run the sweep
python3 summarize.py --sort uch_auroc              # table of everything finished
```

`sample_data/` is a small smoke-test subset (24 slides) — enough to check the
pipeline runs end-to-end, not to reproduce the paper's numbers. To run on the
full dataset instead, point `DISS_DATA` at it before running the same commands:

```bash
export DISS_DATA=/path/to/Dissertation/data
```

Useful flags: `--edges/--aggregators/--backbones` to restrict the grid,
`--stage cv`/`--stage uch` to run only one stage, `--image gated` to add the
image-fusion branch. Interrupted sweeps resume by re-running the same command.
