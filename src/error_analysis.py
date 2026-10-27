"""Phase 5 : exemples visuels de faux positifs et de nodules manqués (scénario B)."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import PATCH_DIR  # noqa: E402

RESULTS = ROOT / "results"


def patch_of(row):
    return np.load(PATCH_DIR / f"{row.seriesuid}.npy")[int(row.candidate_id) - 1].astype(float)


def draw(ax, row, title):
    ax.imshow(patch_of(row)[16], cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main(scenario="B", n=6):
    pred = pd.read_csv(RESULTS / f"predictions_{scenario}.csv")
    feats = pd.read_csv(RESULTS / "features.csv")
    pred = pred.merge(feats[["seriesuid", "candidate_id", "diameter_mm", "sphericity",
                             "skeleton_ratio", "border_dist_mm"]],
                      on=["seriesuid", "candidate_id"])
    fps = pred[pred.label == 0].nlargest(n, "score")
    fns = pred[pred.label == 1].nsmallest(n, "score")
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 5))
    for ax, row in zip(axes[0], fps.itertuples()):
        draw(ax, row, f"FP s={row.score:.2f}\nØ{row.diameter_mm:.0f} mm, sph={row.sphericity:.2f}")
    for ax, row in zip(axes[1], fns.itertuples()):
        draw(ax, row, f"nodule raté s={row.score:.2f}\nØ{row.diameter_mm:.0f} mm, "
                      f"paroi {row.border_dist_mm:.0f} mm")
    for ax in axes.flat:
        ax.axis("off")
    axes[0, 0].text(-0.1, 0.5, "faux positifs", rotation=90, transform=axes[0, 0].transAxes,
                    va="center", ha="right")
    axes[1, 0].text(-0.1, 0.5, "faux négatifs", rotation=90, transform=axes[1, 0].transAxes,
                    va="center", ha="right")
    fig.suptitle(f"Scénario {scenario} : erreurs sur le jeu de test (coupe centrale du patch 32 mm)")
    fig.tight_layout()
    out = RESULTS / "figures" / "erreurs.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("figure :", out)
    detected = pred[pred.label == 1]
    print(f"Positifs détectés par la chaîne morphologique : {len(detected)}")
    print("Score médian des positifs :", round(detected.score.median(), 3),
          "| des négatifs :", round(pred[pred.label == 0].score.median(), 4))


if __name__ == "__main__":
    main(*sys.argv[1:])
