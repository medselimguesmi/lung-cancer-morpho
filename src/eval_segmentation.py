"""Dice et IoU du masque pulmonaire contre les masques de référence LUNA16."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.run_detection import find_scan  # noqa: E402
from src.segmentation import resample_isotropic, segment_lungs  # noqa: E402

LUNG_LABELS = (3, 4)  # poumons gauche et droit (5 = trachée)


def dice_iou(pred, ref):
    inter = np.logical_and(pred, ref).sum()
    union = np.logical_or(pred, ref).sum()
    total = pred.sum() + ref.sum()
    return (2 * inter / total if total else 1.0), (inter / union if union else 1.0)


def main():
    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    luna_dir = ROOT / "data" / "luna16"
    patients = pd.read_csv(ROOT / "results" / "patients.csv")
    rows = []
    for patient in patients.itertuples(index=False):
        image = sitk.ReadImage(str(find_scan(luna_dir / patient.subset, patient.seriesuid)))
        resampled = resample_isotropic(image, config["resample_mm"])
        mask = segment_lungs(
            sitk.GetArrayFromImage(resampled),
            thr=config["thr"], r_open=config["r_open"], r_close=config["r_close"],
        )
        ref_img = sitk.ReadImage(
            str(luna_dir / "seg-lungs-LUNA16" / f"{patient.seriesuid}.mhd")
        )
        ref_res = sitk.Resample(ref_img, resampled, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
        ref = np.isin(sitk.GetArrayFromImage(ref_res), LUNG_LABELS)
        dice, iou = dice_iou(mask, ref)
        rows.append({"seriesuid": patient.seriesuid, "subset": patient.subset,
                     "split": patient.split, "dice": dice, "iou": iou})
        print(f"{patient.seriesuid[-12:]}  Dice={dice:.3f}  IoU={iou:.3f}", flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(ROOT / "results" / "segmentation_scores.csv", index=False)
    print(f"\nDice moyen {scores.dice.mean():.3f}  IoU moyen {scores.iou.mean():.3f}")


if __name__ == "__main__":
    main()
