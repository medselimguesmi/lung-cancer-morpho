"""Phase 5 : métriques de classification, courbes ROC et FROC, comparaison A/B/C."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
SCENARIOS = {"A": "A : morphologie seule", "B": "B : morphologie + ML",
             "C": "C : morphologie + CNN 3D"}


def classification_metrics(y, score, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "sensibilite": tp / (tp + fn) if tp + fn else np.nan,
        "specificite": tn / (tn + fp) if tn + fp else np.nan,
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "auc_roc": roc_auc_score(y, score) if 0 < y.sum() < len(y) else np.nan,
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def covered_nodules(selected, annotations):
    """Indices des nodules annotés dont le rayon contient un candidat sélectionné."""
    ann = annotations[["coordX", "coordY", "coordZ", "diameter_mm"]].to_numpy()
    found = set()
    for row in selected.itertuples():
        d = np.linalg.norm(ann[:, :3] - np.array([row.world_x, row.world_y, row.world_z]),
                           axis=1)
        found.update(np.flatnonzero(d <= ann[:, 3] / 2).tolist())
    return found


def froc_curve(pred, annotations):
    """Sensibilité (sur TOUS les nodules annotés du jeu) vs faux positifs par scan."""
    uids = pred.seriesuid.unique()
    n_scans = len(uids)
    n_nodules = len(annotations[annotations.seriesuid.isin(uids)])
    by_scan = {u: annotations[annotations.seriesuid == u] for u in uids}
    points = []
    for t in sorted(pred.score.unique(), reverse=True):
        sel = pred[pred.score >= t]
        found = sum(len(covered_nodules(g, by_scan[u])) for u, g in sel.groupby("seriesuid"))
        points.append((int((sel.label == 0).sum()) / n_scans, found / n_nodules))
    return np.array(points), n_nodules


def sensitivity_at(points, fp_rate):
    ok = points[points[:, 0] <= fp_rate]
    return float(ok[:, 1].max()) if len(ok) else 0.0


def main():
    annotations = pd.read_csv(ROOT / "data" / "luna16" / "annotations.csv")
    FIGURES.mkdir(exist_ok=True)
    rows, froc_data, roc_data = [], {}, {}
    for key, label in SCENARIOS.items():
        pred = pd.read_csv(RESULTS / f"predictions_{key}.csv")
        y = pred.label.values
        m = classification_metrics(y, pred.score.values, pred.pred.values)
        points, n_nodules = froc_curve(pred, annotations)
        froc_data[key] = points
        roc_data[key] = roc_curve(y, pred.score.values)
        rows.append({"scenario": label, **{k: round(v, 3) if isinstance(v, float) else v
                                           for k, v in m.items()},
                     "sens_a_1FP": round(sensitivity_at(points, 1), 3),
                     "sens_a_4FP": round(sensitivity_at(points, 4), 3),
                     "temps_s": round(float(pred.temps_s.iloc[0]), 2),
                     "donnees": {"A": "aucune (règles)", "B": "descripteurs (~dev)",
                                 "C": "patchs 3D (~dev)"}[key]})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "comparaison_scenarios.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nNodules annotés dans le jeu de test : {n_nodules}")

    fig, ax = plt.subplots(figsize=(5, 5))
    for key, (fpr, tpr, _) in roc_data.items():
        ax.plot(fpr, tpr, label=SCENARIOS[key])
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set(xlabel="1 - spécificité", ylabel="Sensibilité", title="ROC (jeu de test)")
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "roc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for key, pts in froc_data.items():
        ax.step(pts[:, 0], pts[:, 1], where="post", label=SCENARIOS[key])
    ax.set(xlabel="Faux positifs par scan", ylabel="Sensibilité (nodules annotés)",
           title="FROC (jeu de test)", ylim=(0, 1.02))
    ax.legend(loc="lower right")
    fig.savefig(FIGURES / "froc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
