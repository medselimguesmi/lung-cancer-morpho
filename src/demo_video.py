"""Video de demonstration : les etapes de la detection de nodules sur un scanner de test.

Usage : python src/demo_video.py [--patient SERIESUID] [--threshold 0.3]
Sortie : results/demo/video_<id>.mp4 (necessite imageio et imageio-ffmpeg).
Rejoue le vrai pipeline (cache data/cache, find_candidates, predictions du scenario B).
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import yaml
import imageio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache import load_scan  # noqa: E402
from src.detection import find_candidates  # noqa: E402

RESULTS = ROOT / "results"
FPS = 10
BG, FG, ACC = "#0B2540", "#FFFFFF", "#F2994A"


def embed(crop_slice, origin_yx, shape):
    full = np.zeros(shape, dtype=crop_slice.dtype)
    y0, x0 = origin_yx
    full[y0:y0 + crop_slice.shape[0], x0:x0 + crop_slice.shape[1]] = crop_slice
    return full


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient")
    parser.add_argument("--threshold", type=float, default=0.3)
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pred = pd.read_csv(RESULTS / "predictions_B.csv")
    feats = pd.read_csv(RESULTS / "features.csv")
    pred = pred.merge(feats[["seriesuid", "candidate_id", "z_voxel", "y_voxel", "x_voxel",
                             "diameter_mm"]], on=["seriesuid", "candidate_id"])
    pred = pred[pred.split == "test"]
    uid = args.patient or pred[pred.label == 1].sort_values("score").iloc[-1].seriesuid
    cand = pred[pred.seriesuid == uid].reset_index(drop=True)
    subset = pd.read_csv(RESULTS / "patients.csv").set_index("seriesuid").loc[uid].subset
    patient = type("P", (), {"seriesuid": uid, "subset": subset})

    print("chargement du scanner et detection (quelques secondes)...")
    volume, mask, geom = load_scan(patient, config, ROOT / "data" / "luna16")
    spacing = tuple(reversed(geom.GetSpacing()))
    _, inter = find_candidates(
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
    origin = np.asarray(inter["crop_origin_zyx"])
    obj = inter["object_mask"]
    dist = ndi.distance_transform_edt(obj, sampling=spacing)

    pos = cand[cand.label == 1]
    z0 = int(round((pos.iloc[pos.score.argmax()] if len(pos) else cand.iloc[0]).z_voxel))
    z0 = int(np.clip(z0, 0, volume.shape[0] - 1))
    ys, xs = np.nonzero(mask.any(axis=0))
    ylim = (ys.max() + 8, ys.min() - 8)
    xlim = (xs.min() - 8, xs.max() + 8)
    n_kept = int((cand.score >= args.threshold).sum())

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax = fig.add_axes([0.02, 0.04, 0.62, 0.92])
    side = fig.add_axes([0.66, 0.04, 0.32, 0.92]); side.axis("off")

    def local(arr3, z):
        lz = z - origin[0]
        if not 0 <= lz < arr3.shape[0]:
            return np.zeros(volume.shape[1:], dtype=arr3.dtype)
        return embed(arr3[lz], origin[1:], volume.shape[1:])

    def near(z):
        return cand[(cand.z_voxel - z).abs() <= cand.diameter_mm / 2 + 1]

    def render(z, step, title, lines, layer):
        ax.clear(); side.clear(); side.axis("off")
        ax.imshow(volume[z], cmap="gray", vmin=-1000, vmax=400)
        layer(z)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
        side.text(0, 0.97, step, color=ACC, fontsize=15, fontweight="bold", va="top",
                  transform=side.transAxes)
        side.text(0, 0.90, title, color=FG, fontsize=21, fontweight="bold", va="top",
                  wrap=True, transform=side.transAxes)
        side.text(0, 0.70, "\n\n".join(lines), color="#CADCFC", fontsize=14, va="top",
                  wrap=True, transform=side.transAxes)
        side.text(0, 0.02, f"patient ...{uid[-12:]}   coupe z={z}", color="#8CA0B3", fontsize=11,
                  transform=side.transAxes)
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()

    def no_layer(z):
        pass

    def lung_layer(z):
        ax.contour(mask[z], levels=[0.5], colors="cyan", linewidths=1.5)

    def objects_layer(z):
        m = local(obj, z)
        ax.imshow(np.ma.masked_where(~m, m), cmap="autumn", alpha=0.6)

    def distance_layer(z):
        d = local(dist, z)
        ax.imshow(np.ma.masked_where(d == 0, d), cmap="viridis", alpha=0.85, vmin=0, vmax=8)

    def ws_layer(z):
        l = local(inter["watershed_labels"], z)
        ax.imshow(np.ma.masked_where(l == 0, l % 20), cmap="nipy_spectral", alpha=0.6)

    def skel_layer(z):
        s = ndi.binary_dilation(local(inter["skeleton"], z), iterations=2)
        ax.imshow(np.ma.masked_where(~s, s), cmap="autumn", alpha=1.0)

    def cand_layer(z):
        n = near(z)
        ax.plot(n.x_voxel, n.y_voxel, "r+", ms=14, mew=2)

    def decision_layer(z):
        for r in near(z).itertuples():
            keep = r.score >= args.threshold
            ax.add_patch(plt.Circle((r.x_voxel, r.y_voxel), max(r.diameter_mm / 2, 4), fill=False,
                                    ec="lime" if keep else "red", lw=3 if keep else 1.2))
            ax.text(r.x_voxel + 7, r.y_voxel - 7, f"{r.score:.2f}", color="yellow", fontsize=11)
            if r.label == 1:
                ax.plot(r.x_voxel, r.y_voxel, "w*", ms=18)

    frames = []

    def hold(img, seconds):
        frames.extend([img] * int(seconds * FPS))

    # 0 titre
    ax.clear(); side.clear(); side.axis("off"); ax.axis("off")
    fig.text(0.5, 0.62, "Détection de nodules pulmonaires", ha="center", color=FG, fontsize=36,
             fontweight="bold")
    t1 = fig.text(0.5, 0.5, "par morphologie mathématique", ha="center", color=FG, fontsize=30)
    t2 = fig.text(0.5, 0.36, f"Démonstration sur un scanner de test LUNA16 (…{uid[-12:]})",
                  ha="center", color=ACC, fontsize=18)
    fig.canvas.draw()
    hold(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(), 3)
    for t in list(fig.texts):
        t.remove()

    # 1 defilement du scanner
    for z in range(max(0, z0 - 30), min(volume.shape[0], z0 + 31), 3):
        frames.append(render(z, "ÉTAPE 1", "Scanner thoracique", [
            "Volume 3D rééchantillonné à 1 mm, en unités Hounsfield.",
            "Défilement des coupes axiales : poumons, vaisseaux, os."], no_layer))
    stages = [
        ("ÉTAPE 2", "Segmentation des poumons", lung_layer, 4, [
            "Seuil −320 HU, suppression du bord, ouverture (sphère r=2),",
            "fermeture (disque r=8) et remplissage des trous.",
            "Dice moyen 0,974 sur 25 patients."]),
        ("ÉTAPE 3", "Objets denses", objects_layer, 4, [
            "Masque poumon ET HU ≥ −320 :",
            "vaisseaux, nodules et paroi apparaissent."]),
        ("ÉTAPE 4", "Transformée de distance", distance_layer, 4, [
            "Distance de chaque voxel au fond (mm).",
            "Le centre d'un objet compact (nodule) est un maximum ;",
            "les vaisseaux fins restent faibles."]),
        ("ÉTAPE 5", "Watershed sur la distance", ws_layer, 4, [
            "Marqueurs = maxima locaux de la distance.",
            "Le watershed sépare les nodules collés à un vaisseau",
            "ou à la paroi ; troncature en boule autour du pic."]),
        ("ÉTAPE 6", "Squelette et filtre de forme", skel_layer, 4, [
            "Squelette 3D de chaque région :",
            "longueur et branchements. On filtre aussi",
            "le diamètre (3 à 30 mm)."]),
        ("ÉTAPE 7", f"{len(cand)} candidats", cand_layer, 4, [
            "Chaque croix = un candidat de la chaîne morphologique.",
            "Presque tous sont de faux positifs (vaisseaux, hile, paroi)."]),
        ("ÉTAPE 8", "Décision du scénario B", decision_layer, 5, [
            f"Random Forest sur 17 descripteurs ; vert = score ≥ {args.threshold}.",
            f"{n_kept} candidat(s) retenu(s) sur {len(cand)}.",
            "Étoile blanche = nodule annoté par les radiologues."]),
    ]
    for step, title, layer, secs, lines in stages:
        hold(render(z0, step, title, lines, layer), secs)

    # 9 defilement final avec decisions
    for z in range(max(0, z0 - 24), min(volume.shape[0], z0 + 25), 2):
        frames.append(render(z, "RÉSULTAT", "Candidats et décisions", [
            "Défilement du volume : cercles verts = retenus, rouges = rejetés,",
            "avec le score du Random Forest."], decision_layer))
    hold(frames[-1], 2)

    out = RESULTS / "demo" / f"video_{uid[-12:]}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out), fps=FPS, codec="libx264", quality=7,
                            macro_block_size=1) as writer:
        for f in frames:
            writer.append_data(f)
    print(f"video : {out}  ({len(frames) / FPS:.0f} s)")
    print(f"candidats : {len(cand)} | retenus : {n_kept} | "
          f"candidats positifs retenus : {int(((cand.label == 1) & (cand.score >= args.threshold)).sum())}/"
          f"{int((cand.label == 1).sum())}")


if __name__ == "__main__":
    main()
