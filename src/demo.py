"""Demonstration : un scanner de test, son masque pulmonaire, ses candidats et la decision du scenario B.

Usage : python src/demo.py [--patient SERIESUID] [--scenario B]
Sans --patient, prend le patient de test dont le nodule a le meilleur score.
Utilise le cache data/cache et les predictions deja calculees (execution en quelques secondes).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache import load_scan  # noqa: E402

RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient")
    parser.add_argument("--scenario", default="B")
    parser.add_argument("--threshold", type=float, default=0.3, help="seuil de decision sur le score")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pred = pd.read_csv(RESULTS / f"predictions_{args.scenario}.csv")
    feats = pd.read_csv(RESULTS / "features.csv")
    pred = pred.merge(feats[["seriesuid", "candidate_id", "z_voxel", "y_voxel", "x_voxel",
                             "diameter_mm", "sphericity"]], on=["seriesuid", "candidate_id"])
    pred = pred[pred.split == "test"]
    uid = args.patient or pred[pred.label == 1].sort_values("score").iloc[-1].seriesuid
    cand = pred[pred.seriesuid == uid].reset_index(drop=True)
    patient = pd.read_csv(RESULTS / "patients.csv").set_index("seriesuid").loc[uid]
    patient = patient.rename("p")
    patient = type("P", (), {"seriesuid": uid, "subset": patient.subset})
    volume, mask, _ = load_scan(patient, config, ROOT / "data" / "luna16")

    pos = cand[cand.label == 1]
    z = int(round((pos.iloc[pos.score.argmax()] if len(pos) else cand.iloc[0]).z_voxel))
    near = cand[(cand.z_voxel - z).abs() <= cand.diameter_mm / 2 + 1]
    kept = cand[cand.score >= args.threshold]

    fig, ax = plt.subplots(1, 4, figsize=(20, 5.4))
    ct = volume[z]
    ax[0].imshow(ct, cmap="gray", vmin=-1000, vmax=400); ax[0].set_title(f"1. Scanner (HU), coupe z={z}")
    ax[1].imshow(ct, cmap="gray", vmin=-1000, vmax=400)
    ax[1].contour(mask[z], levels=[0.5], colors="cyan", linewidths=1)
    ax[1].set_title("2. Segmentation des poumons")
    ax[2].imshow(ct, cmap="gray", vmin=-1000, vmax=400)
    ax[2].plot(near.x_voxel, near.y_voxel, "r+", ms=9)
    ax[2].set_title(f"3. Candidats morphologiques\n({len(cand)} dans le volume, {len(near)} sur cette coupe)")
    ax[3].imshow(ct, cmap="gray", vmin=-1000, vmax=400)
    for r in near.itertuples():
        keep = r.score >= args.threshold
        ax[3].add_patch(plt.Circle((r.x_voxel, r.y_voxel), max(r.diameter_mm / 2, 4), fill=False,
                        ec="lime" if keep else "red", lw=2 if keep else 1))
        ax[3].text(r.x_voxel + 6, r.y_voxel - 6, f"{r.score:.2f}", color="yellow", fontsize=9)
        if r.label == 1:
            ax[3].plot(r.x_voxel, r.y_voxel, "w*", ms=14)
    ax[3].set_title(f"4. Decision scenario {args.scenario} (vert = nodule, score >= {args.threshold})\n"
                    f"etoile blanche = nodule annote ; {len(kept)} retenus / {len(cand)}")
    for a in ax:
        a.axis("off")
    # recadrage sur les poumons
    ys, xs = np.nonzero(mask[z])
    for a in ax:
        a.set_xlim(xs.min() - 10, xs.max() + 10); a.set_ylim(ys.max() + 10, ys.min() - 10)
    fig.suptitle(f"Demo - patient {uid[-12:]}", fontsize=12)
    fig.tight_layout()
    out = RESULTS / "demo" / f"demo_{uid[-12:]}.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("figure :", out)
    print(f"candidats : {len(cand)} | retenus (score>={args.threshold}) : {len(kept)} | "
          f"candidats positifs retenus : {int(((cand.label == 1) & (cand.score >= args.threshold)).sum())}/"
          f"{int((cand.label == 1).sum())}")


if __name__ == "__main__":
    main()
