import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detection import (  # noqa: E402
    evaluate_candidates,
    find_candidates,
    morphological_gradient,
    white_tophat,
)
from src.segmentation import resample_isotropic, segment_lungs  # noqa: E402


def find_scan(scan_dir, seriesuid):
    matches = list((scan_dir).rglob(f"{seriesuid}.mhd"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Scan attendu une fois, trouvé {len(matches)} : {seriesuid}")
    return matches[0]


def save_diagnostic(volume, image, intermediates, candidates, annotations, output_path, config):
    origin = np.asarray(intermediates["crop_origin_zyx"])
    annotation_indices = [
        image.TransformPhysicalPointToContinuousIndex(
            (row.coordX, row.coordY, row.coordZ)
        )
        for row in annotations.itertuples(index=False)
    ]
    if annotation_indices:
        z = int(round(annotation_indices[0][2]))
    elif candidates:
        z = int(round(candidates[0]["z_voxel"]))
    else:
        z = volume.shape[0] // 2

    local_z = z - int(origin[0])
    if not 0 <= local_z < intermediates["object_mask"].shape[0]:
        local_z = intermediates["object_mask"].shape[0] // 2
        z = int(origin[0]) + local_z
    local_volume = volume[z]
    object_slice = intermediates["object_mask"][local_z]
    label_slice = intermediates["watershed_labels"][local_z]
    skeleton_slice = intermediates["skeleton"][local_z]

    def place_crop_on_slice(cropped):
        full_slice = np.zeros(volume.shape[1:], dtype=cropped.dtype)
        y_start, x_start = origin[1:]
        full_slice[y_start:y_start + cropped.shape[0],
                   x_start:x_start + cropped.shape[1]] = cropped
        return full_slice

    object_slice = place_crop_on_slice(object_slice)
    label_slice = place_crop_on_slice(label_slice)
    skeleton_slice = place_crop_on_slice(skeleton_slice)
    gradient = morphological_gradient(local_volume, config["morphology_gradient_radius"])
    tophat = white_tophat(local_volume, config["morphology_tophat_radius"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes[0, 0].imshow(local_volume, cmap="gray", vmin=-1000, vmax=400)
    axes[0, 0].contour(object_slice, levels=[0.5], colors="cyan", linewidths=0.7)
    axes[0, 0].set_title(f"CT et objets, z={z}")
    axes[0, 1].imshow(gradient, cmap="magma")
    axes[0, 1].set_title("Gradient morphologique")
    axes[0, 2].imshow(tophat, cmap="magma")
    axes[0, 2].set_title("Top-hat blanc")
    axes[1, 0].imshow(local_volume, cmap="gray", vmin=-1000, vmax=400)
    axes[1, 0].imshow(np.ma.masked_where(label_slice == 0, label_slice), cmap="nipy_spectral", alpha=0.55)
    axes[1, 0].set_title("Régions du watershed")
    axes[1, 1].imshow(local_volume, cmap="gray", vmin=-1000, vmax=400)
    axes[1, 1].imshow(np.ma.masked_where(~skeleton_slice, skeleton_slice), cmap="autumn", alpha=0.9)
    axes[1, 1].set_title("Squelettes retenus")
    axes[1, 2].imshow(local_volume, cmap="gray", vmin=-1000, vmax=400)
    axes[1, 2].set_title(f"Candidats : {len(candidates)}")
    for candidate in candidates:
        if abs(candidate["z_voxel"] - z) <= max(1, candidate["diameter_mm"] / 2):
            axes[1, 2].plot(candidate["x_voxel"], candidate["y_voxel"], "r+", ms=8)
    spacing_xyz = image.GetSpacing()
    for row, index_xyz in zip(annotations.itertuples(index=False), annotation_indices):
        if abs(index_xyz[2] - z) <= max(1, row.diameter_mm / (2 * spacing_xyz[2])):
            axes[1, 2].add_patch(Circle(
                (index_xyz[0], index_xyz[1]),
                radius=row.diameter_mm / (2 * spacing_xyz[0]),
                fill=False,
                edgecolor="lime",
                linewidth=1.2,
            ))
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def process_scan(patient, annotations, config, luna_dir, figure_dir):
    scan_path = find_scan(luna_dir / patient.subset, patient.seriesuid)
    image = sitk.ReadImage(str(scan_path))
    resampled = resample_isotropic(image, config["resample_mm"])
    volume = sitk.GetArrayFromImage(resampled)
    mask = segment_lungs(
        volume,
        thr=config["thr"],
        r_open=config["r_open"],
        r_close=config["r_close"],
    )
    spacing_zyx = tuple(reversed(resampled.GetSpacing()))
    started = time.time()
    candidates, intermediates = find_candidates(
        volume,
        mask,
        spacing=spacing_zyx,
        intensity_threshold=config["candidate_intensity_threshold"],
        min_seed_radius_mm=config["candidate_min_seed_radius_mm"],
        min_distance_mm=config["candidate_min_distance_mm"],
        min_diameter_mm=config["candidate_min_diameter_mm"],
        max_diameter_mm=config["candidate_max_diameter_mm"],
        max_skeleton_ratio=config["candidate_max_skeleton_ratio"],
        max_branch_points=config["candidate_max_branch_points"],
        opening_radius=config.get("candidate_opening_radius", 0),
        truncate_factor=config.get("candidate_truncate_factor", 0.0),
        return_intermediates=True,
    )
    elapsed = time.time() - started
    for candidate in candidates:
        index_xyz = (
            candidate["x_voxel"], candidate["y_voxel"], candidate["z_voxel"]
        )
        physical_xyz = resampled.TransformContinuousIndexToPhysicalPoint(index_xyz)
        candidate.update({
            "seriesuid": patient.seriesuid,
            "subset": patient.subset,
            "coordX": physical_xyz[0],
            "coordY": physical_xyz[1],
            "coordZ": physical_xyz[2],
        })

    scan_annotations = annotations[annotations.seriesuid == patient.seriesuid]
    annotation_values = scan_annotations[
        ["coordX", "coordY", "coordZ", "diameter_mm"]
    ].to_numpy()
    metrics = evaluate_candidates(
        candidates,
        annotation_values,
        resampled.TransformContinuousIndexToPhysicalPoint,
    )
    metrics.update({
        "seriesuid": patient.seriesuid,
        "subset": patient.subset,
        "split": patient.split,
        "temps_detection_s": elapsed,
    })

    if len(intermediates["object_mask"]):
        figure_path = figure_dir / f"{patient.seriesuid}.png"
        save_diagnostic(
            volume, resampled, intermediates, candidates, scan_annotations, figure_path, config
        )
    return candidates, metrics


def main():
    parser = argparse.ArgumentParser(description="Détection morphologique des candidats LUNA16")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--limit", type=int, help="nombre maximal de scans à traiter")
    args = parser.parse_args()

    with (ROOT / "config.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    luna_dir = ROOT / "data" / "luna16"
    patients = pd.read_csv(ROOT / "results" / "patients.csv")
    annotations = pd.read_csv(luna_dir / "annotations.csv")
    if args.split != "all":
        patients = patients[patients.split == args.split]
    if args.limit is not None:
        patients = patients.head(args.limit)

    output_dir = ROOT / "results"
    figure_dir = output_dir / "detection_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_candidates = []
    all_metrics = []
    for index, patient in enumerate(patients.itertuples(index=False), start=1):
        print(f"[{index}/{len(patients)}] {patient.seriesuid}", flush=True)
        candidates, metrics = process_scan(
            patient, annotations, config, luna_dir, figure_dir
        )
        all_candidates.extend(candidates)
        all_metrics.append(metrics)
        print(
            f"  candidats={metrics['candidats']}, rappel={metrics['rappel']:.3f}, "
            f"FP/scan={metrics['faux_positifs_par_scan']}, "
            f"temps={metrics['temps_detection_s']:.1f}s",
            flush=True,
        )

    candidate_columns = [
        "seriesuid", "subset", "candidate_id", "coordX", "coordY", "coordZ",
        "z_voxel", "y_voxel", "x_voxel", "volume_mm3", "diameter_mm",
        "max_radius_mm", "skeleton_length_mm", "branch_points", "mean_hu",
    ]
    pd.DataFrame(all_candidates, columns=candidate_columns).to_csv(
        output_dir / "candidates.csv", index=False
    )
    metrics_frame = pd.DataFrame(all_metrics)
    metrics_frame.to_csv(output_dir / "detection_eval.csv", index=False)
    n_annotations = int(metrics_frame["nodules"].sum()) if len(metrics_frame) else 0
    total_matches = int(metrics_frame["nodules_retrouves"].sum()) if len(metrics_frame) else 0
    total_scans = len(metrics_frame)
    print(
        f"\nRappel global : {total_matches}/{n_annotations} "
        f"({total_matches / n_annotations:.3f})" if n_annotations else "\nAucune annotation évaluée"
    )
    if total_scans:
        print(f"Faux positifs par scan : {metrics_frame.faux_positifs.mean():.2f}")
    print(f"Résultats : {output_dir / 'candidates.csv'}")
    print(f"Figures : {figure_dir}")


if __name__ == "__main__":
    main()