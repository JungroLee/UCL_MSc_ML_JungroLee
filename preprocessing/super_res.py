"""
Step 1/3 — x4 super-resolution (Swin2SR) of the patch tiles.

299 px tiles -> ~1196 px, so HoVer-Net sees them at ~40x pixel density. The
SAME model / settings are used on TCGA and UCH tiles, which is what lets a fixed
radius (see build_graphs.py) mean the same physical neighbourhood in both
cohorts.

Model loaded once, run in fp16, tiles in batches (batch=8 fits in 8 GB; 16 OOMs).
Resume-safe: tiles already present in --out are skipped.

    python3 super_res.py --in <tiles_10x> --out <tiles_sr>
    python3 super_res.py --in <tiles_10x> --out <tiles_sr> --batch 8
"""
import argparse, os, glob
import numpy as np
import cv2
import torch
from transformers import Swin2SRForImageSuperResolution, AutoImageProcessor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "caidas/swin2SR-classical-sr-x4-64"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", required=True, help="folder of *.png tiles")
    ap.add_argument("--out", dest="outdir", required=True, help="where to write SR tiles")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    half = DEVICE == "cuda"
    print(f"device={DEVICE} half={half} batch={args.batch} | loading Swin2SR x4 ...")
    proc = AutoImageProcessor.from_pretrained(MODEL)
    model = Swin2SRForImageSuperResolution.from_pretrained(MODEL).eval().to(DEVICE)
    if half:
        model = model.half()

    pngs = sorted(glob.glob(os.path.join(args.indir, "*.png")))
    todo = [p for p in pngs if not os.path.exists(os.path.join(args.outdir, os.path.basename(p)))]
    print(f"{len(pngs)} tiles ({len(todo)} to do, {len(pngs) - len(todo)} already done)")

    done = 0
    for i in range(0, len(todo), args.batch):
        batch_paths = todo[i:i + args.batch]
        imgs = []
        for p in batch_paths:
            bgr = cv2.imread(p)
            imgs.append(None if bgr is None else cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        keep = [(p, im) for p, im in zip(batch_paths, imgs) if im is not None]
        if not keep:
            continue
        paths, arrs = zip(*keep)
        pixel = proc(list(arrs), return_tensors="pt")["pixel_values"].to(DEVICE)
        if half:
            pixel = pixel.half()
        with torch.no_grad():
            recon = model(pixel_values=pixel).reconstruction    # [B,3,H,W]
        recon = recon.clamp(0, 1).float().cpu().numpy()
        for j, p in enumerate(paths):
            sr = (np.transpose(recon[j], (1, 2, 0)) * 255.0).round().astype(np.uint8)
            cv2.imwrite(os.path.join(args.outdir, os.path.basename(p)),
                        cv2.cvtColor(sr, cv2.COLOR_RGB2BGR))
            done += 1
        if done % 400 < args.batch:
            print(f"  {done}/{len(todo)} done")
    print(f"\ndone={done} -> {args.outdir}")


if __name__ == "__main__":
    main()
