"""
Collect every finished sweep cell into one table:

    edge x aggregator x backbone  ->  TCGA 3-fold AUROC  +  UCH external AUROC

Usage
    python3 summarize.py                      # every cohort found
    python3 summarize.py --cohort random32
    python3 summarize.py --sort uch_auroc     # rank by external performance

Writes results/summary_<cohort>.csv and prints a markdown table.
"""
import os
import csv
import json
import argparse

import paths


def parse_tag(tag):
    """`<edge>_<aggregator>_<backbone>`, where the aggregator may contain '_'."""
    for agg in paths.AGGREGATORS:
        for bk in paths.BACKBONES:
            suffix = f"_{agg}_{bk}"
            if tag.endswith(suffix):
                return {"edge": tag[:-len(suffix)], "aggregator": agg, "backbone": bk}
    return {"edge": tag, "aggregator": "", "backbone": ""}


def read_cell(cohort, tag):
    """One config: the CV metrics and the UCH metrics, either of which may be absent."""
    base = os.path.join(paths.RUNS, cohort, tag)
    row = {"config": tag}
    row.update(**parse_tag(tag))

    cv_dir = os.path.join(base, "cv")
    if os.path.isdir(cv_dir):
        for sub in sorted(os.listdir(cv_dir)):
            p = os.path.join(cv_dir, sub, "metrics.json")
            if os.path.exists(p):
                m = json.load(open(p))
                row.update(cv_mean_auc=m.get("mean_auc"), cv_std_auc=m.get("std_auc"),
                           cv_pooled_auc=m.get("pooled_auc"), cv_spearman=m.get("spearman"),
                           n_patients=m.get("n_patients"))
                break

    p = os.path.join(base, "uch", "uch_metrics.json")
    if os.path.exists(p):
        m = json.load(open(p))
        row.update(uch_auroc=m.get("auroc"), uch_spearman=m.get("spearman_rs"),
                   uch_sensitivity=m.get("sensitivity"),
                   uch_specificity=m.get("specificity"), uch_n=m.get("n"))
    return row


FIELDS = ["config", "edge", "aggregator", "backbone",
          "cv_mean_auc", "cv_std_auc", "cv_pooled_auc", "cv_spearman", "n_patients",
          "uch_auroc", "uch_spearman", "uch_sensitivity", "uch_specificity", "uch_n"]


def fmt(v, nd=4):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def summarize(cohort, sort_key):
    root = os.path.join(paths.RUNS, cohort)
    if not os.path.isdir(root):
        print(f"no runs for cohort '{cohort}' under {root}")
        return
    rows = [read_cell(cohort, t) for t in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, t))]
    rows = [r for r in rows if r.get("cv_mean_auc") is not None
            or r.get("uch_auroc") is not None]
    if not rows:
        print(f"cohort '{cohort}': no finished runs yet")
        return
    rows.sort(key=lambda r: (r.get(sort_key) is None, -(r.get(sort_key) or 0)))

    out_csv = os.path.join(paths.OUT, f"summary_{cohort}.csv")
    os.makedirs(paths.OUT, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"\n## {cohort}  ({len(rows)} configs, sorted by {sort_key})\n")
    print("| edge | aggregator | backbone | TCGA 3-fold AUROC | TCGA pooled | UCH AUROC | UCH Spearman |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        mean, std = r.get("cv_mean_auc"), r.get("cv_std_auc")
        cv = f"{mean:.4f} ± {std:.4f}" if mean is not None and std is not None else fmt(mean)
        print(f"| {r['edge']} | {r['aggregator']} | {r['backbone']} | {cv} | "
              f"{fmt(r.get('cv_pooled_auc'))} | {fmt(r.get('uch_auroc'))} | "
              f"{fmt(r.get('uch_spearman'))} |")
    print(f"\nwritten to {out_csv}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", default=None, choices=sorted(paths.COHORTS))
    ap.add_argument("--sort", default="cv_mean_auc",
                    choices=["cv_mean_auc", "cv_pooled_auc", "uch_auroc"])
    a = ap.parse_args()
    for cohort in ([a.cohort] if a.cohort else sorted(paths.COHORTS)):
        summarize(cohort, a.sort)


if __name__ == "__main__":
    main()
