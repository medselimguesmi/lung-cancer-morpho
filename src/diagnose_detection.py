"""Pourquoi chaque nodule annoté est-il retrouvé ou manqué ?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache import load_scan  # noqa: E402
from src.detection import find_candidates  # noqa: E402


def main(split="dev"):
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    luna_dir = ROOT / "data" / "luna16"
    patients = pd.read_csv(ROOT / "results" / "patients.csv")
    patients = patients[patients.split == split]
    annotations = pd.read_csv(luna_dir / "annotations.csv")
    rows = []
    for patient in patients.itertuples(index=False):
        volume, mask, geom = load_scan(patient, config, luna_dir)
        spacing = tuple(reversed(geom.GetSpacing()))
        candidates, inter = find_candidates(
            volume, mask, spacing=spacing,
            intensity_threshold=config["candidate_intensity_threshold"],
            min_seed_radius_mm=config["candidate_min_seed_radius_mm"],
            min_distance_mm=config["candidate_min_distance_mm"],
            min_diameter_mm=config["candidate_min_diameter_mm"],
            max_diameter_mm=config["candidate_max_diameter_mm"],
            max_skeleton_ratio=config["candidate_max_skeleton_ratio"],
            max_branch_points=config["candidate_max_branch_points"],
            opening_radius=config.get("candidate_opening_radius", 0),
            truncate_factor=config.get("candidate_truncate_factor", 0.0),
            return_intermediates=True)
        origin = np.array(inter["crop_origin_zyx"])
        rejected = {r["label"]: r for r in inter["rejected"]}
        for a in annotations[annotations.seriesuid == patient.seriesuid].itertuples():
            idx = geom.TransformPhysicalPointToContinuousIndex((a.coordX, a.coordY, a.coordZ))
            zyx = np.rint(idx[::-1]).astype(int)
            in_lung = bool(mask[tuple(zyx)])
            local = zyx - origin
            ok = (local >= 0).all() and (local < inter["object_mask"].shape).all()
            in_obj = bool(ok and inter["object_mask"][tuple(local)])
            label = int(inter["watershed_labels"][tuple(local)]) if ok else 0
            # un candidat accepté couvre-t-il le centre ?
            dists = [np.linalg.norm((np.array([c["x_voxel"], c["y_voxel"], c["z_voxel"]])
                                     - idx) * geom.GetSpacing()) for c in candidates]
            near = min(dists) if dists else np.inf
            if near <= a.diameter_mm / 2:
                status = "retrouvé"
            elif not in_lung:
                status = "hors masque"
            elif not in_obj:
                status = "sous le seuil d'intensité"
            elif label == 0:
                status = "sans graine"
            elif label in rejected:
                status = "rejeté:" + rejected[label]["reason"]
            else:
                status = "région acceptée mais centre éloigné"
            rows.append({"seriesuid": patient.seriesuid[-8:], "diam_nodule": a.diameter_mm,
                         "statut": status, "dist_candidat_proche_mm": round(near, 1),
                         "region": rejected.get(label, {})})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 90)
    print(df.to_string())
    print(df.statut.value_counts())


if __name__ == "__main__":
    main(*sys.argv[1:])
