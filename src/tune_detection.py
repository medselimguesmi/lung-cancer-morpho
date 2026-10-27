"""Banc d'essai des paramètres de détection sur le split dev (scans en cache)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache import load_scan  # noqa: E402
from src.detection import evaluate_candidates, find_candidates  # noqa: E402


def detect(volume, mask, geom, cfg):
    return find_candidates(
        volume, mask, spacing=tuple(reversed(geom.GetSpacing())),
        intensity_threshold=cfg["candidate_intensity_threshold"],
        min_seed_radius_mm=cfg["candidate_min_seed_radius_mm"],
        min_distance_mm=cfg["candidate_min_distance_mm"],
        min_diameter_mm=cfg["candidate_min_diameter_mm"],
        max_diameter_mm=cfg["candidate_max_diameter_mm"],
        max_skeleton_ratio=cfg["candidate_max_skeleton_ratio"],
        max_branch_points=cfg["candidate_max_branch_points"],
        opening_radius=cfg.get("candidate_opening_radius", 0),
        truncate_factor=cfg.get("candidate_truncate_factor", 0.0),
        use_watershed=cfg.get("use_watershed", True))


def run_config(cfg, patients, annotations, luna_dir):
    found = total = fps = cands = 0
    started = time.time()
    for p in patients.itertuples(index=False):
        volume, mask, geom = load_scan(p, cfg, luna_dir)
        c = detect(volume, mask, geom, cfg)
        ann = annotations[annotations.seriesuid == p.seriesuid][
            ["coordX", "coordY", "coordZ", "diameter_mm"]].to_numpy()
        m = evaluate_candidates(c, ann, geom.TransformContinuousIndexToPhysicalPoint)
        found += m["nodules_retrouves"]; total += m["nodules"]
        fps += m["faux_positifs"]; cands += m["candidats"]
    n = len(patients)
    return {"nodules": total, "retrouves": found, "rappel": round(found / total, 3),
            "faux_positifs_par_scan": round(fps / n, 1), "candidats_par_scan": round(cands / n, 1),
            "temps_s": round(time.time() - started)}


def main(grid, split="dev"):
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    luna_dir = ROOT / "data" / "luna16"
    patients = pd.read_csv(ROOT / "results" / "patients.csv")
    patients = patients[patients.split == split]
    annotations = pd.read_csv(luna_dir / "annotations.csv")
    out = ROOT / "results" / "tests_detection.csv"
    rows = []
    for overrides in grid:
        cfg = {**base, **overrides}
        res = run_config(cfg, patients, annotations, luna_dir)
        row = {**{k.replace("candidate_", ""): v for k, v in overrides.items()}, **res}
        print(row, flush=True)
        rows.append(row)
    df = pd.DataFrame(rows)
    if out.exists():
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    T, D, O = "candidate_intensity_threshold", "candidate_min_distance_mm", "candidate_opening_radius"
    S, B = "candidate_max_skeleton_ratio", "candidate_max_branch_points"
    R = "candidate_min_seed_radius_mm"
    F = "candidate_truncate_factor"
    grid = [
        {F: 1.3, D: 8.0, R: 1.2, S: 3.0, B: 3},
        {F: 1.3, D: 8.0, R: 1.2, S: 3.0, B: 3, T: -400},
        {F: 1.6, D: 8.0, R: 1.2, S: 3.0, B: 3, T: -400},
    ]
    main(grid)
