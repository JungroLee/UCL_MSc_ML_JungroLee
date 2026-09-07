"""
Sweep every combination of

    edge construction  x  aggregation method  x  GNN backbone

on one TCGA patch bag (random-32 or annotated-32), and score each combination
twice:

    stage 1   TCGA 3-fold CV        paper_eval.py    -> metrics.json
    stage 2   train on ALL of TCGA, infer on UCH
                                    external_uch.py  -> uch_metrics.json

Both stages use the paper's protocol: patient-grouped folds, held-out HR+/HER2-
patients, pooled AUROC. Stage 2 is the only place real clinical OncotypeDX
recurrence scores are used, so it is the honest external number.

Usage
    python3 run_sweep.py --cohort random32
    python3 run_sweep.py --cohort annotated32 --edges knn6 delaunay
    python3 run_sweep.py --cohort random32 --aggregators mean --backbones gcn sage
    python3 run_sweep.py --cohort random32 --image gated       # + UNI2-h branch
    python3 run_sweep.py --cohort random32 --stage cv          # skip UCH
    python3 run_sweep.py --cohort random32 --dry_run           # just list the jobs

Every job is skipped if its output already exists, so the sweep is resume-safe:
re-run the same command after an interruption and it picks up where it stopped.
"""
import os
import sys
import json
import time
import argparse
import itertools
import subprocess

import paths

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen hyper-parameters. These are the winner of the earlier 144-config grid,
# held fixed so that a difference between two cells of this sweep is a
# difference in edges / aggregation / backbone and nothing else.
HP = ["--task", "regression",
      "--hidden", "64",
      "--lr", "1e-4",
      "--gnn_layers", "2",
      "--dropout", "0.3",
      "--weight_decay", "1e-3",
      "--batch_size", "1",
      "--max_patches", "32",
      "--epochs", "80"]


def _hp(name):
    return HP[HP.index(name) + 1]


def cv_subdir(agg, backbone):
    """The folder paper_eval.py creates inside its --out_dir."""
    return (f"{_hp('--task')}_{agg}_{backbone}"
            f"_h{_hp('--hidden')}_lr{float(_hp('--lr')):g}")


def build_jobs(a):
    """One (tag, stage, cmd) per config x stage, skipping finished work."""
    jobs = []
    edges = []
    for edge in a.edges:
        if os.path.isdir(paths.graph_dir(a.cohort, edge)):
            edges.append(edge)
        else:
            print(f"[skip] {a.cohort} has no '{edge}' graphs "
                  f"({paths.graph_dir(a.cohort, edge)})")
    for edge, agg, bk in itertools.product(edges, a.aggregators, a.backbones):
        graph_dir = paths.graph_dir(a.cohort, edge)

        tag = f"{edge}_{agg}_{bk}"
        common = HP + ["--aggregator", agg, "--gnn_type", bk]
        if a.image != "none":
            common += ["--img_mode", a.image]
        img_tcga = ["--fm_emb", paths.fm_emb(a.cohort)] if a.image != "none" else []

        if a.stage in ("cv", "both"):
            out = os.path.join(paths.RUNS, a.cohort, tag, "cv")
            # paper_eval.py appends its own <task>_<agg>_<gnn>_h..._lr... subdir
            done = os.path.join(out, cv_subdir(agg, bk), "metrics.json")
            if not os.path.exists(done):
                jobs.append((tag, "cv", ["python3", "-u", "paper_eval.py"] + common
                             + img_tcga
                             + ["--graph_dir", graph_dir,
                                "--labels_csv", paths.LABELS,
                                "--clinical_csv", paths.CLINICAL,
                                "--num_workers", "2",
                                "--out_dir", out]))

        if a.stage in ("uch", "both"):
            out = os.path.join(paths.RUNS, a.cohort, tag, "uch")
            uch_graph = paths.uch_graph_dir(edge)
            if not os.path.isdir(uch_graph):
                print(f"[skip] uch/{edge}: no graphs at {uch_graph}")
            elif not os.path.exists(os.path.join(out, "uch_metrics.json")):
                img_uch = (["--tcga_fm", paths.fm_emb(a.cohort),
                            "--uch_fm", paths.UCH_FM] if a.image != "none" else [])
                jobs.append((tag, "uch", ["python3", "-u", "external_uch.py"] + common
                             + img_uch
                             + ["--tcga_graph", graph_dir,
                                "--tcga_labels", paths.LABELS,
                                "--clinical", paths.CLINICAL,
                                "--uch_graph", uch_graph,
                                "--uch_labels", paths.UCH_LABELS,
                                "--out_dir", out]))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default="random32", choices=sorted(paths.COHORTS),
                    help="which TCGA patch bag to train on")
    ap.add_argument("--edges", nargs="+", default=sorted(paths.EDGES),
                    choices=sorted(paths.EDGES))
    ap.add_argument("--aggregators", nargs="+", default=paths.AGGREGATORS,
                    choices=paths.AGGREGATORS)
    ap.add_argument("--backbones", nargs="+", default=paths.BACKBONES,
                    choices=paths.BACKBONES)
    ap.add_argument("--image", default="none", choices=["none", "concat", "gated", "only"],
                    help="UNI2-h image branch: none = cell graph alone (default), "
                         "gated = the best model's fusion")
    ap.add_argument("--stage", default="both", choices=["cv", "uch", "both"])
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    jobs = build_jobs(a)
    edges = [e for e in a.edges if os.path.isdir(paths.graph_dir(a.cohort, e))]
    n_cfg = len(edges) * len(a.aggregators) * len(a.backbones)
    print(f"cohort {a.cohort} | {n_cfg} configs "
          f"({len(edges)} edges x {len(a.aggregators)} aggregators x "
          f"{len(a.backbones)} backbones) | image branch: {a.image}")
    print(f"{len(jobs)} jobs to run (finished ones skipped)\n")

    if a.dry_run:
        for tag, stage, cmd in jobs:
            print(f"  {tag:28s} {stage:4s}  {' '.join(cmd)}")
        return

    os.makedirs(paths.LOGS, exist_ok=True)
    failed = []
    for i, (tag, stage, cmd) in enumerate(jobs, 1):
        log = os.path.join(paths.LOGS, f"{a.cohort}_{tag}_{stage}.log")
        print(f"[{i}/{len(jobs)}] {tag} {stage} -> {log}", flush=True)
        t0 = time.time()
        with open(log, "w") as f:
            rc = subprocess.call(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if rc == 0:
            print(f"        done in {mins:.1f} min", flush=True)
        else:
            print(f"        FAILED rc={rc} after {mins:.1f} min -- see the log", flush=True)
            failed.append((tag, stage, log))

    print(f"\nfinished: {len(jobs) - len(failed)}/{len(jobs)}")
    for tag, stage, log in failed:
        print(f"  failed: {tag} {stage}  {log}")
    print(f"\nresults under {os.path.join(paths.RUNS, a.cohort)}")
    print("summarise with:  python3 summarize.py --cohort " + a.cohort)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
