"""Phase 4 : descripteurs, étiquettes et patchs 3D des candidats."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import marching_cubes, mesh_surface_area

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cache import load_scan  # noqa: E402
from src.detection import find_candidates, morphological_gradient  # noqa: E402

PATCH = 32
PATCH_DIR = ROOT / "data" / "patches"
FEATURE_COLUMNS = [
    "volume_mm3", "diameter_mm", "sphericity", "solidity", "elongation",
    "mean_hu", "std_hu", "grad_mean", "glcm_contrast", "glcm_homogeneity",
    "glcm_energy", "glcm_correlation", "skeleton_length_mm", "branch_points",
    "max_radius_mm", "skeleton_ratio", "border_dist_mm",
]


def shape_descriptors(region, spacing):
    """Sphéricité, solidité et élongation d'une région binaire 3D."""
    voxel = float(np.prod(spacing))
    volume = region.sum() * voxel
    padded = np.pad(region, 1).astype(np.float32)
    try:
        verts, faces, _, _ = marching_cubes(padded, 0.5, spacing=spacing)
        area = mesh_surface_area(verts, faces)
        sphericity = float(np.pi ** (1 / 3) * (6 * volume) ** (2 / 3) / area)
    except (ValueError, RuntimeError):
        sphericity = np.nan
    points = np.argwhere(region) * np.asarray(spacing)
    try:
        solidity = float(volume / ConvexHull(points).volume)
    except (QhullError, ValueError):
        solidity = np.nan
    if len(points) > 3:
        eig = np.sort(np.linalg.eigvalsh(np.cov(points.T)))
        elongation = float(np.sqrt(eig[-1] / max(eig[0], 1e-6)))
    else:
        elongation = np.nan
    return min(sphericity, 1.5), min(solidity, 1.0), elongation


def glcm_descriptors(patch_hu):
    """Texture GLCM sur la coupe centrale (HU quantifiés sur 16 niveaux)."""
    sl = patch_hu[patch_hu.shape[0] // 2]
    q = np.clip((sl + 1000) / 1400 * 15, 0, 15).astype(np.uint8)
    g = graycomatrix(q, distances=[1], angles=[0, np.pi / 2], levels=16,
                     symmetric=True, normed=True)
    return {f"glcm_{k}": float(graycoprops(g, k).mean())
            for k in ("contrast", "homogeneity", "energy", "correlation")}


def extract_patch(volume, center_zyx):
    """Patch 32^3 (HU clippés à [-1000, 400], normalisés 0-1) centré sur le candidat."""
    half = PATCH // 2
    c = np.rint(center_zyx).astype(int)
    pad = np.pad(volume, half, constant_values=-1000)
    z, y, x = c
    patch = pad[z:z + PATCH, y:y + PATCH, x:x + PATCH]
    return ((np.clip(patch, -1000, 400) + 1000) / 1400).astype(np.float16)


def label_candidates(frame, annotations, geom):
    """1 si le centre du candidat est dans le rayon d'un nodule annoté."""
    ann = annotations[["coordX", "coordY", "coordZ", "diameter_mm"]].to_numpy()
    labels = []
    for row in frame.itertuples():
        w = np.array(geom.TransformContinuousIndexToPhysicalPoint(
            (row.x_voxel, row.y_voxel, row.z_voxel)))
        hit = len(ann) and (np.linalg.norm(ann[:, :3] - w, axis=1) <= ann[:, 3] / 2).any()
        labels.append(int(bool(hit)))
    return labels


def scan_features(patient, annotations, cfg, luna_dir, save_patches=True):
    volume, mask, geom = load_scan(patient, cfg, luna_dir)
    spacing = tuple(reversed(geom.GetSpacing()))
    cands, inter = find_candidates(
        volume, mask, spacing=spacing,
        intensity_threshold=cfg["candidate_intensity_threshold"],
        min_seed_radius_mm=cfg["candidate_min_seed_radius_mm"],
        min_distance_mm=cfg["candidate_min_distance_mm"],
        min_diameter_mm=cfg["candidate_min_diameter_mm"],
        max_diameter_mm=cfg["candidate_max_diameter_mm"],
        max_skeleton_ratio=cfg["candidate_max_skeleton_ratio"],
        max_branch_points=cfg["candidate_max_branch_points"],
        opening_radius=cfg.get("candidate_opening_radius", 0),
        truncate_factor=cfg.get("candidate_truncate_factor", 0.0),
        return_intermediates=True)
    if not cands:
        return pd.DataFrame(), np.zeros((0, PATCH, PATCH, PATCH), np.float16)
    origin = np.array(inter["crop_origin_zyx"])
    objects = ndi.find_objects(inter["watershed_labels"])
    border = ndi.distance_transform_edt(mask, sampling=spacing)
    crop_vol = volume[tuple(slice(o, o + s) for o, s in zip(origin, inter["object_mask"].shape))]
    grad = morphological_gradient(crop_vol.astype(np.float32), cfg["morphology_gradient_radius"])
    rows, patches = [], []
    for c in cands:
        sl = objects[c["label_id"] - 1]
        region = inter["watershed_labels"][sl] == c["label_id"]
        sph, sol, elong = shape_descriptors(region, spacing)
        hu = crop_vol[sl][region].astype(np.float32)
        edge = region & ~ndi.binary_erosion(region)
        center = np.array([c["z_voxel"], c["y_voxel"], c["x_voxel"]])
        patch = extract_patch(volume, center)
        row = {k: c[k] for k in ("candidate_id", "z_voxel", "y_voxel", "x_voxel",
                                 "volume_mm3", "diameter_mm", "max_radius_mm",
                                 "skeleton_length_mm", "branch_points", "mean_hu")}
        row.update({
            "sphericity": sph, "solidity": sol, "elongation": elong,
            "std_hu": float(hu.std()),
            "grad_mean": float(grad[sl][edge].mean()) if edge.any() else 0.0,
            "skeleton_ratio": c["skeleton_length_mm"] / max(c["diameter_mm"], 1e-6),
            "border_dist_mm": float(border[tuple(np.rint(center).astype(int))]),
            **glcm_descriptors(patch.astype(np.float32) * 1400 - 1000),
        })
        rows.append(row)
        patches.append(patch)
    frame = pd.DataFrame(rows)
    scan_ann = annotations[annotations.seriesuid == patient.seriesuid]
    frame["label"] = label_candidates(frame, scan_ann, geom)
    frame.insert(0, "seriesuid", patient.seriesuid)
    frame.insert(1, "split", patient.split)
    frame["world_x"], frame["world_y"], frame["world_z"] = zip(*[
        geom.TransformContinuousIndexToPhysicalPoint((r.x_voxel, r.y_voxel, r.z_voxel))
        for r in frame.itertuples()])
    return frame, np.stack(patches)


def main():
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    luna_dir = ROOT / "data" / "luna16"
    patients = pd.read_csv(ROOT / "results" / "patients.csv")
    annotations = pd.read_csv(luna_dir / "annotations.csv")
    PATCH_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, p in enumerate(patients.itertuples(index=False), 1):
        frame, patches = scan_features(p, annotations, cfg, luna_dir)
        np.save(PATCH_DIR / f"{p.seriesuid}.npy", patches)
        frames.append(frame)
        print(f"[{i}/{len(patients)}] {p.split} {len(frame)} candidats, "
              f"{int(frame.label.sum()) if len(frame) else 0} positifs", flush=True)
    features = pd.concat(frames, ignore_index=True)
    features.to_csv(ROOT / "results" / "features.csv", index=False)
    n_pos = int(features.label.sum())
    print(f"\n{len(features)} candidats, {n_pos} positifs "
          f"({100 * n_pos / len(features):.1f} %) -> results/features.csv")


if __name__ == "__main__":
    main()
