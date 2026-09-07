"""
Build (or rebuild) the small `sample_data/` folder checked into this repo, so
the package runs end-to-end on a fresh clone without the full ~20GB dataset.

Picks a fixed set of 24 TCGA slides (12 recurrence-positive / 12 negative, all
distinct patients, HR+/HER2-, 4 per CV3_odx85_mip fold x label -- chosen so
every fold's validation split in both paper_eval.py and external_uch.py
contains both labels, which AUROC-based model selection needs to be
well-defined on a sample this small) plus a matching 12-slide UCH subset, and
copies only their patch graphs + matching fm_emb rows + label/clinical rows
into sample_data/, mirroring the layout paths.py expects.

This is a demo/smoke-test fixture, NOT a way to reproduce the paper's numbers
-- 24 slides is far too small for a meaningful AUROC.

Usage (reads from the full dataset, writes into this repo):
    export DISS_DATA=/path/to/Dissertation/data
    python3 scripts/make_sample_data.py
"""
import os
import csv
import glob
import shutil

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.environ.get("DISS_DATA")
if not SRC:
    raise SystemExit(
        "Set DISS_DATA to the full dataset root first, e.g.\n"
        "  export DISS_DATA=/path/to/Dissertation/data")
DST = os.path.join(HERE, "sample_data")

TCGA_SLIDES = [
    # 24 slides, distinct patients, HR+/HER2- (the paper's eval subset), 4 per
    # (CV3_odx85_mip fold x label). Chosen by search so that BOTH the
    # paper_eval.py per-fold val carve (val_frac=0.15, seed=42) and the
    # external_uch.py all-TCGA val carve (val_frac=0.1, seed=42) land on a
    # val set containing both labels -- required for AUROC-based model
    # selection to be defined (non-NaN) on a sample this small.
    "TCGA-A7-A6VX-01Z-00-DX1.F74DA243-C65A-4997-BCA0-F1C89675978C",
    "TCGA-D8-A1JH-01Z-00-DX1.4A4F2502-612C-421D-9F64-444BF2C85620",
    "TCGA-BH-A1ES-01Z-00-DX1.C54C809F-748F-4BB0-B018-A8A83A4134C0",
    "TCGA-BH-A18P-01Z-00-DX1.C66642D3-BE44-4D65-B4AE-C1C3D959D22C",
    "TCGA-C8-A27A-01Z-00-DX1.0E26C46D-CD65-40F3-8976-EB4415582934",
    "TCGA-BH-A0HU-01Z-00-DX1.73B38904-E4F8-4F45-BD75-A27EC833B6DE",
    "TCGA-D8-A13Y-01Z-00-DX1.02321E77-A11E-41A5-95FE-BB897EA5CE58",
    "TCGA-BH-A5IZ-01Z-00-DX1.6C871030-82E1-463E-A67B-976A5F3DCDB2",
    "TCGA-AO-A1KO-01Z-00-DX1.EEB5E0A0-92B2-42CD-9F7A-00E9250B561F",
    "TCGA-A2-A3KD-01Z-00-DX1.E400D529-A0D1-4C78-AD5C-33EB0508B128",
    "TCGA-AN-A0XL-01Z-00-DX1.E90AA056-51DC-4A6B-96EB-A0B707496912",
    "TCGA-A2-A0CU-01Z-00-DX1.5B77D21C-A497-478B-9752-3730322AD9ED",
    "TCGA-LL-A8F5-01Z-00-DX1.7579425E-1425-4160-A9DD-3D50F4C5428D",
    "TCGA-A2-A0D4-01Z-00-DX1.35E43827-DABC-4685-A62A-333672923349",
    "TCGA-AN-A0AM-01Z-00-DX1.169CE39A-DD54-46D8-8D03-60B69A473CDB",
    "TCGA-E9-A1NE-01Z-00-DX1.f332124a-cdab-4ab5-82d1-4b8b7f3c9821",
    "TCGA-E2-A106-01Z-00-DX1.A8D49C06-3C93-48A6-87F8-86D646BEA28C",
    "TCGA-E2-A10B-01Z-00-DX1.148CFC4B-EE65-4A7E-918F-10C72F37CB0F",
    "TCGA-AR-A24L-01Z-00-DX1.218A0AE1-D070-4A16-A277-31185F10724D",
    "TCGA-B6-A0IB-01Z-00-DX1.BAA1D655-1B80-49E2-B1EB-2ECC83DED989",
    "TCGA-A8-A092-01Z-00-DX1.2A55EA0C-47CD-426B-9697-F2B761730585",
    "TCGA-E2-A1II-01Z-00-DX1.7F782477-0F92-4C57-8735-A1E3F95A6B94",
    "TCGA-AR-A2LK-01Z-00-DX1.FBD59C38-CD4E-4C22-BC74-A57C192A9BBC",
    "TCGA-A8-A07U-01Z-00-DX1.69D356C3-C7FC-47E9-B753-BE421263343F",
]

UCH_SLIDES = [
    "UCH_BRCA_RS_1", "UCH_BRCA_RS_105", "UCH_BRCA_RS_106",
    "UCH_BRCA_RS_11", "UCH_BRCA_RS_115", "UCH_BRCA_RS_117",
    "UCH_BRCA_RS_10", "UCH_BRCA_RS_100", "UCH_BRCA_RS_101",
    "UCH_BRCA_RS_102", "UCH_BRCA_RS_103", "UCH_BRCA_RS_107",
]


def copy_graphs(cohort_dir, edge_subdir, slide_stems, dst_dir):
    """Copy every patch .pt whose filename starts with one of slide_stems."""
    src = os.path.join(cohort_dir, "graphs_v2", edge_subdir)
    if not os.path.isdir(src):
        print(f"  [skip] no {src}")
        return 0
    os.makedirs(dst_dir, exist_ok=True)
    all_paths = glob.glob(os.path.join(src, "**", "*.pt"), recursive=True)
    n = 0
    for p in all_paths:
        stem = os.path.basename(p).split("__")[0]
        if stem in slide_stems:
            shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
            n += 1
    return n


def subset_fm_emb(src_path, slide_stems, dst_path):
    """fm_emb caches are keyed per PATCH (stem includes __x..._y...), so keep
    every row whose patch stem starts with one of our slide stems."""
    if not os.path.exists(src_path):
        print(f"  [skip] no {src_path}")
        return 0
    blob = torch.load(src_path, weights_only=False)
    keep = [i for i, s in enumerate(blob["stems"])
            if s.split("__")[0] in slide_stems]
    if not keep:
        print(f"  [skip] no matching rows in {src_path}")
        return 0
    new_blob = {
        "stems": [blob["stems"][i] for i in keep],
        "emb": blob["emb"][keep],
    }
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save(new_blob, dst_path)
    return len(keep)


def subset_csv(src_csv, id_col, slide_stems, dst_csv, patient_fallback=False):
    """Copy only rows whose id matches a wanted slide (or, if patient_fallback,
    the 12-char patient barcode of a wanted slide)."""
    patients = {s[:12] for s in slide_stems} if patient_fallback else set()
    with open(src_csv, newline="") as f:
        r = csv.DictReader(f)
        rows = [row for row in r
                if row[id_col] in slide_stems or row[id_col] in patients]
        fieldnames = r.fieldnames
    os.makedirs(os.path.dirname(dst_csv), exist_ok=True)
    with open(dst_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(
            f"DISS_DATA not found: {SRC}\n"
            "Set DISS_DATA to the full dataset before running this script.")
    os.makedirs(DST, exist_ok=True)

    # ---- TCGA cell graphs: random32 (Another_dataset) + annotated32 ----
    for cohort_name, edge_subdirs in [
        ("Another_dataset", ["feat_morph", "feat_delaunay", "feat_radius110"]),
        ("Annotated_Dataset", ["feat_morph", "feat_delaunay"]),
    ]:
        cohort_src = os.path.join(SRC, cohort_name)
        for edge_subdir in edge_subdirs:
            dst_dir = os.path.join(DST, cohort_name, "graphs_v2", edge_subdir)
            n = copy_graphs(cohort_src, edge_subdir, set(TCGA_SLIDES), dst_dir)
            print(f"{cohort_name}/{edge_subdir}: copied {n} patch graphs")

        # fm_emb (UNI2-h) for this cohort
        fm_src = os.path.join(cohort_src, "fm_emb", "uni2h_grid4.pt")
        fm_dst = os.path.join(DST, cohort_name, "fm_emb", "uni2h_grid4.pt")
        n = subset_fm_emb(fm_src, set(TCGA_SLIDES), fm_dst)
        print(f"{cohort_name}/fm_emb: kept {n} patch rows")

    # ---- UCH cell graphs (uch_graphs_v2/<edge>/... , no graphs_v2 subdir) ----
    for edge_subdir in ["feat_morph", "feat_delaunay", "feat_radius110"]:
        src_dir = os.path.join(SRC, "uch_graphs_v2", edge_subdir)
        dst_dir = os.path.join(DST, "uch_graphs_v2", edge_subdir)
        if not os.path.isdir(src_dir):
            print(f"  [skip] no {src_dir}")
            continue
        os.makedirs(dst_dir, exist_ok=True)
        n = 0
        for p in glob.glob(os.path.join(src_dir, "**", "*.pt"), recursive=True):
            stem = os.path.basename(p).split("__")[0]
            if stem in UCH_SLIDES:
                shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
                n += 1
        print(f"uch_graphs_v2/{edge_subdir}: copied {n} patch graphs")

    fm_src = os.path.join(SRC, "uch_fm_emb", "uni2h_grid4.pt")
    fm_dst = os.path.join(DST, "uch_fm_emb", "uni2h_grid4.pt")
    n = subset_fm_emb(fm_src, set(UCH_SLIDES), fm_dst)
    print(f"uch_fm_emb: kept {n} patch rows")

    # ---- labels / clinical CSVs ----
    n = subset_csv(os.path.join(SRC, "labels.csv"), "slide_id",
                    set(TCGA_SLIDES), os.path.join(DST, "labels.csv"))
    print(f"labels.csv: kept {n} rows")

    n = subset_csv(os.path.join(SRC, "tcga_brca_complete.csv"), "slide",
                    set(TCGA_SLIDES), os.path.join(DST, "tcga_brca_complete.csv"),
                    patient_fallback=True)
    print(f"tcga_brca_complete.csv: kept {n} rows")

    n = subset_csv(os.path.join(SRC, "uch_labels.csv"), "slide_id",
                    set(UCH_SLIDES), os.path.join(DST, "uch_labels.csv"))
    print(f"uch_labels.csv: kept {n} rows")

    print("\nDone. Sample data at:", DST)


if __name__ == "__main__":
    main()
