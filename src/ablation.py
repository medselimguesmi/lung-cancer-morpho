"""Phase 5 : étude d'ablation de la chaîne morphologique."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache import load_scan  # noqa: E402
from src.classify import load_features, run_scenario_b  # noqa: E402
from src.detection import evaluate_candidates  # noqa: E402
from src.features import FEATURE_COLUMNS  # noqa: E402
from src.metrics import classification_metrics, froc_curve, sensitivity_at  # noqa: E402
from src.tune_detection import detect  # noqa: E402

RESULTS = ROOT / "results"
BIG = 10 ** 9

# Étapes de la chaîne complète ; True = repose sur la morphologie mathématique.
CHAIN = [
    ("rééchantillonnage + fenêtrage HU", False),
    ("seuillage du poumon", False),
    ("ouverture (segmentation)", True),
    ("fermeture + remplissage des trous", True),
    ("seuillage des objets denses", False),
    ("transformée de distance + marqueurs", True),
    ("watershed", True),
    ("troncature en boule autour du pic", True),
    ("squelette + filtre de forme", True),
    ("gradient morphologique (descripteur)", True),
    ("descripteurs de texture (GLCM) et d'intensité", False),
    ("classification (règles / ML / CNN)", False),
]

DETECTION_VARIANTS = {
    "chaîne complète": ({}, None),
    "sans ouverture (segmentation)": ({"r_open": 0}, "ouverture (segmentation)"),
    "sans fermeture": ({"r_close": 0}, "fermeture + remplissage des trous"),
    "sans watershed": ({"use_watershed": False}, "watershed"),
    "sans troncature": ({"candidate_truncate_factor": 0.0}, "troncature en boule autour du pic"),
    "sans squelette": ({"candidate_max_skeleton_ratio": BIG,
                        "candidate_max_branch_points": BIG}, "squelette + filtre de forme"),
}

FEATURE_VARIANTS = {
    "tous les descripteurs": [],
    "sans gradient": ["grad_mean"],
    "sans squelette": ["skeleton_length_mm", "branch_points", "skeleton_ratio", "max_radius_mm"],
    "sans descripteurs de forme": ["sphericity", "solidity", "elongation"],
}


def detection_ablation(base):
    luna_dir = ROOT / "data" / "luna16"
    patients = pd.read_csv(RESULTS / "patients.csv")
    annotations = pd.read_csv(luna_dir / "annotations.csv")
    rows = []
    for name, (overrides, _) in DETECTION_VARIANTS.items():
        cfg = {**base, **overrides}
        acc = {s: {"n": 0, "found": 0, "fp": 0, "scans": 0} for s in ("dev", "test")}
        for p in patients.itertuples(index=False):
            volume, mask, geom = load_scan(p, cfg, luna_dir)
            cands = detect(volume, mask, geom, cfg)
            ann = annotations[annotations.seriesuid == p.seriesuid][
                ["coordX", "coordY", "coordZ", "diameter_mm"]].to_numpy()
            m = evaluate_candidates(cands, ann, geom.TransformContinuousIndexToPhysicalPoint)
            a = acc[p.split]
            a["n"] += m["nodules"]; a["found"] += m["nodules_retrouves"]
            a["fp"] += m["faux_positifs"]; a["scans"] += 1
        row = {"variante": name}
        for s, a in acc.items():
            row[f"rappel_{s}"] = round(a["found"] / a["n"], 3)
            row[f"fp_par_scan_{s}"] = round(a["fp"] / a["scans"], 1)
        rows.append(row)
        print(row, flush=True)
    frame = pd.DataFrame(rows)
    ref = frame.iloc[0]
    frame["delta_rappel_test"] = (frame.rappel_test - ref.rappel_test).round(3)
    frame["delta_fp_test"] = (frame.fp_par_scan_test - ref.fp_par_scan_test).round(1)
    frame["etape_retiree"] = [v[1] for v in DETECTION_VARIANTS.values()]
    frame.to_csv(RESULTS / "ablation_detection.csv", index=False)
    return frame


def feature_ablation():
    dev, test = load_features()
    annotations = pd.read_csv(ROOT / "data" / "luna16" / "annotations.csv")
    rows = []
    for name, dropped in FEATURE_VARIANTS.items():
        columns = [c for c in FEATURE_COLUMNS if c not in dropped]
        run_scenario_b(dev, test, columns=columns, tag="abl", verbose=False)
        pred = pd.read_csv(RESULTS / "predictions_abl.csv")
        m = classification_metrics(pred.label.values, pred.score.values, pred.pred.values)
        points, _ = froc_curve(pred, annotations)
        rows.append({"variante": name, "n_descripteurs": len(columns),
                     "auc_test": round(m["auc_roc"], 3),
                     "sens_a_4FP": round(sensitivity_at(points, 4), 3),
                     "sens_a_10FP": round(sensitivity_at(points, 10), 3)})
        print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame["delta_auc"] = (frame.auc_test - frame.auc_test.iloc[0]).round(3)
    frame.to_csv(RESULTS / "ablation_descripteurs.csv", index=False)
    return frame


def usage_share():
    morph = sum(1 for _, m in CHAIN if m)
    share = pd.DataFrame(CHAIN, columns=["etape", "morphologie"])
    share.to_csv(RESULTS / "ablation_chaine.csv", index=False)
    print(f"\nÉtapes reposant sur la morphologie : {morph}/{len(CHAIN)} "
          f"= {100 * morph / len(CHAIN):.0f} %")
    return morph, len(CHAIN)


if __name__ == "__main__":
    base = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    print(detection_ablation(base).to_string(index=False))
    print(feature_ablation().to_string(index=False))
    usage_share()
