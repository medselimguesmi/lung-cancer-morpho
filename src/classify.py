"""Phase 4 : scénarios A (règles), B (morphologie + ML) et C (morphologie + CNN 3D).

Découpage par patient : le split `dev` sert à l'entraînement et à la validation
(validation croisée groupée par patient), le split `test` n'est utilisé qu'une
fois, pour les prédictions finales.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import FEATURE_COLUMNS, PATCH_DIR  # noqa: E402

RESULTS = ROOT / "results"
SEED = 0


def load_features():
    frame = pd.read_csv(RESULTS / "features.csv")
    frame[FEATURE_COLUMNS] = frame[FEATURE_COLUMNS].fillna(frame[FEATURE_COLUMNS].median())
    return (frame[frame.split == "dev"].reset_index(drop=True),
            frame[frame.split == "test"].reset_index(drop=True))


def save_predictions(name, frame, score, threshold, seconds):
    out = frame[["seriesuid", "candidate_id", "split", "label",
                 "world_x", "world_y", "world_z"]].copy()
    out["score"] = score
    out["pred"] = (score >= threshold).astype(int)
    out["scenario"] = name
    out["temps_s"] = seconds
    out.to_csv(RESULTS / f"predictions_{name}.csv", index=False)


# ---------------------------------------------------------------- scénario A
RULE_GRID = {
    "sphericity": [0.5, 0.6, 0.7, 0.8],          # sphéricité minimale
    "solidity": [0.7, 0.8, 0.9],                  # solidité minimale
    "skeleton_ratio": [1.0, 1.5, 2.0, 3.0],       # squelette court (max)
    "border_dist_mm": [0, 5],                     # 0 = nodules juxta-pleuraux admis
}


def rule_score(frame, rules):
    ok = np.stack([
        frame.sphericity >= rules["sphericity"],
        frame.solidity >= rules["solidity"],
        frame.skeleton_ratio <= rules["skeleton_ratio"],
        frame.border_dist_mm >= rules["border_dist_mm"],
    ], axis=1)
    return ok.all(axis=1).astype(float) + 0.01 * frame.sphericity.clip(0, 1)


def fit_rules(dev, max_fp_per_scan=10.0):
    """Choisit les seuils sur dev : sensibilité maximale sous un budget de FP/scan."""
    n_scans = dev.seriesuid.nunique()
    best, best_key = None, (-1, 0)
    for values in itertools.product(*RULE_GRID.values()):
        rules = dict(zip(RULE_GRID, values))
        pred = rule_score(dev, rules) >= 1
        tp = int((pred & (dev.label == 1)).sum())
        fp = int((pred & (dev.label == 0)).sum())
        if fp / n_scans <= max_fp_per_scan and (tp, -fp) > best_key:
            best, best_key = rules, (tp, -fp)
    return best


def run_scenario_a(dev, test):
    started = time.time()
    rules = fit_rules(dev)
    score = rule_score(test, rules)
    save_predictions("A", test, score.values, 1.0, time.time() - started)
    print("Scénario A, seuils retenus :", rules)
    return rules


# ---------------------------------------------------------------- scénario B
def model_zoo(pos_weight):
    return {
        "SVM": [make_pipeline(StandardScaler(), SVC(
            C=c, gamma="scale", probability=True, class_weight="balanced",
            random_state=SEED)) for c in (0.3, 1, 3, 10)],
        "RandomForest": [RandomForestClassifier(
            n_estimators=300, max_depth=d, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
            for d in (3, 5, None)],
        "XGBoost": [XGBClassifier(
            n_estimators=200, max_depth=d, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=pos_weight, eval_metric="logloss",
            random_state=SEED, n_jobs=1) for d in (2, 3, 4)],
    }


def cv_score(model, X, y, groups, n_splits=5):
    """Précision moyenne (aire PR) en validation croisée groupée par patient."""
    scores = []
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if y[tr].sum() == 0 or y[va].sum() == 0:
            continue
        model.fit(X[tr], y[tr])
        scores.append(average_precision_score(y[va], model.predict_proba(X[va])[:, 1]))
    return float(np.mean(scores)) if scores else 0.0


def run_scenario_b(dev, test, columns=FEATURE_COLUMNS, tag="B", verbose=True):
    X, y, groups = dev[columns].values, dev.label.values, dev.seriesuid.values
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    summary, best = [], {}
    for family, models in model_zoo(pos_weight).items():
        scored = [(cv_score(m, X, y, groups), i) for i, m in enumerate(models)]
        ap, idx = max(scored)
        model = models[idx]
        started = time.time()
        model.fit(X, y)
        proba = model.predict_proba(test[columns].values)[:, 1]
        seconds = time.time() - started
        summary.append({"modele": family, "ap_cv_dev": round(ap, 3),
                        "auc_test": round(roc_auc_score(test.label, proba), 3)})
        best[family] = (ap, model, proba, seconds)
        save_predictions(f"{tag}_{family}", test, proba, 0.5, seconds)
    top = max(best, key=lambda k: best[k][0])      # choisi sur la validation, pas sur test
    save_predictions(tag, test, best[top][2], 0.5, best[top][3])
    if verbose:
        print(pd.DataFrame(summary).to_string(index=False))
        print(f"Scénario {tag} : meilleur modèle en validation = {top}")
    if tag == "B":
        imp = pd.Series(best["RandomForest"][1].feature_importances_, index=columns)
        imp.sort_values(ascending=False).to_csv(
            RESULTS / "importance_descripteurs.csv", header=["importance"])
    return top


# ---------------------------------------------------------------- scénario C
def load_patches(frame):
    cache, patches = {}, []
    for uid, cid in zip(frame.seriesuid, frame.candidate_id):
        if uid not in cache:
            cache[uid] = np.load(PATCH_DIR / f"{uid}.npy")
        patches.append(cache[uid][cid - 1])
    return np.stack(patches).astype(np.float32)[:, None]


def run_scenario_c(dev, test, epochs=40):
    import torch
    from torch import nn
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    uids = dev.seriesuid.unique()
    rng.shuffle(uids)
    val_uids = set(uids[: max(2, len(uids) // 4)])
    is_val = dev.seriesuid.isin(val_uids).values
    Xd, yd = load_patches(dev), dev.label.values.astype(np.float32)
    Xt = load_patches(test)
    Xtr, ytr, Xva, yva = Xd[~is_val], yd[~is_val], Xd[is_val], yd[is_val]

    def block(i, o):
        return nn.Sequential(nn.Conv3d(i, o, 3, padding=1), nn.BatchNorm3d(o), nn.ReLU(),
                             nn.MaxPool3d(2))
    net = nn.Sequential(block(1, 8), block(8, 16), block(16, 32), nn.AdaptiveAvgPool3d(1),
                        nn.Flatten(), nn.Dropout(0.3), nn.Linear(32, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor(float((ytr == 0).sum() / max((ytr == 1).sum(), 1)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def augment(batch):
        for axis in (2, 3, 4):
            if rng.random() < 0.5:
                batch = np.flip(batch, axis=axis)
        k = int(rng.integers(0, 4))
        return np.ascontiguousarray(np.rot90(batch, k, axes=(3, 4)))

    def predict(X):
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(torch.from_numpy(X))).numpy().ravel()

    curves, best_auc, best_state = [], -1.0, None
    started = time.time()
    for epoch in range(1, epochs + 1):
        net.train()
        order = rng.permutation(len(Xtr))
        running = 0.0
        for start in range(0, len(order), 32):
            idx = order[start:start + 32]
            xb = torch.from_numpy(augment(Xtr[idx]))
            yb = torch.from_numpy(ytr[idx])[:, None]
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(idx)
        auc = roc_auc_score(yva, predict(Xva)) if 0 < yva.sum() < len(yva) else 0.5
        curves.append({"epoch": epoch, "loss_train": running / len(Xtr), "auc_val": auc})
        if auc > best_auc:
            best_auc, best_state = auc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    proba = predict(Xt)
    pd.DataFrame(curves).to_csv(RESULTS / "courbes_cnn.csv", index=False)
    save_predictions("C", test, proba, 0.5, time.time() - started)
    print(f"Scénario C : meilleur AUC validation {best_auc:.3f}, "
          f"AUC test {roc_auc_score(test.label, proba):.3f}")


def main():
    dev, test = load_features()
    print(f"dev : {len(dev)} candidats ({int(dev.label.sum())} positifs, "
          f"{dev.seriesuid.nunique()} patients) ; test : {len(test)} candidats "
          f"({int(test.label.sum())} positifs, {test.seriesuid.nunique()} patients)")
    run_scenario_a(dev, test)
    run_scenario_b(dev, test)
    run_scenario_c(dev, test)


if __name__ == "__main__":
    main()
