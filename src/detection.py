import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from skimage.morphology import ball, skeletonize
from skimage.segmentation import watershed


def morphological_gradient(image, radius=1):
    """Gradient morphologique d'une image 2D ou 3D."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("image doit être un tableau 2D ou 3D")
    if radius < 1:
        raise ValueError("radius doit être supérieur ou égal à 1")
    footprint = ndi.generate_binary_structure(image.ndim, image.ndim)
    footprint = ndi.iterate_structure(footprint, radius)
    return (ndi.grey_dilation(image, footprint=footprint)
            - ndi.grey_erosion(image, footprint=footprint))


def white_tophat(image, radius=3):
    """Chapeau blanc pour faire ressortir les petites structures claires."""
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError("image doit être un tableau 2D ou 3D")
    if radius < 1:
        raise ValueError("radius doit être supérieur ou égal à 1")
    footprint = ndi.generate_binary_structure(image.ndim, image.ndim)
    footprint = ndi.iterate_structure(footprint, radius)
    values = image.astype(np.float32, copy=False)
    return values - ndi.grey_opening(values, footprint=footprint)


def _truncate_to_ball(region, distance, spacing, factor):
    """Garde la partie de la région dans une boule centrée sur son pic de distance.

    Rayon de la boule = factor x rayon du pic : coupe les vaisseaux ou la paroi
    attachés au nodule avant le filtrage par squelette.
    """
    masked = np.where(region, distance, -1.0)
    peak = np.array(np.unravel_index(np.argmax(masked), region.shape))
    radius_mm = factor * float(masked[tuple(peak)])
    half = np.ceil(radius_mm / spacing).astype(int) + 1
    lo = np.maximum(peak - half, 0)
    hi = np.minimum(peak + half + 1, region.shape)
    box = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    grid = np.indices(region[box].shape).astype(np.float32)
    offsets = (grid - (peak - lo).reshape(3, 1, 1, 1)) * spacing.reshape(3, 1, 1, 1)
    inside = (offsets ** 2).sum(axis=0) <= radius_mm ** 2
    truncated = np.zeros_like(region)
    truncated[box] = region[box] & inside
    return truncated


def find_candidates(
    volume,
    lung_mask,
    spacing=(1.0, 1.0, 1.0),
    intensity_threshold=-600,
    min_seed_radius_mm=1.5,
    min_distance_mm=6.0,
    min_diameter_mm=3.0,
    max_diameter_mm=30.0,
    max_skeleton_ratio=2.5,
    max_branch_points=3,
    opening_radius=0,
    use_watershed=True,
    truncate_factor=0.0,
    return_intermediates=False,
):
    """Détecte des régions candidates dans un volume HU et un masque pulmonaire.

    Les tableaux d'entrée sont en ordre (z, y, x). `spacing` suit le même ordre
    et est exprimé en millimètres. Retourne une liste de dictionnaires, ou un
    couple (candidats, intermédiaires) si return_intermediates vaut True.
    """
    volume = np.asarray(volume)
    lung_mask = np.asarray(lung_mask, dtype=bool)
    spacing = np.asarray(spacing, dtype=float)
    if volume.ndim != 3 or lung_mask.shape != volume.shape:
        raise ValueError("volume et lung_mask doivent avoir la même forme 3D")
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError("spacing doit contenir trois valeurs positives (z, y, x)")
    if min_diameter_mm <= 0 or max_diameter_mm < min_diameter_mm:
        raise ValueError("les diamètres minimum et maximum sont incohérents")

    points = np.argwhere(lung_mask)
    if not len(points):
        empty = []
        intermediates = {
            "object_mask": np.zeros((0, 0, 0), dtype=bool),
            "watershed_labels": np.zeros((0, 0, 0), dtype=np.int32),
            "skeleton": np.zeros((0, 0, 0), dtype=bool),
            "crop_origin_zyx": (0, 0, 0),
        }
        return (empty, intermediates) if return_intermediates else empty

    lower = points.min(axis=0)
    upper = points.max(axis=0) + 1
    crop = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper))
    cropped_volume = volume[crop]
    cropped_lung = lung_mask[crop]
    object_mask = cropped_lung & (cropped_volume >= intensity_threshold)
    if opening_radius > 0:  # détache les nodules des vaisseaux fins
        object_mask = ndi.binary_opening(object_mask, structure=ball(opening_radius))
    if not object_mask.any():
        labels = np.zeros(object_mask.shape, dtype=np.int32)
        candidates = []
    else:
        distance = ndi.distance_transform_edt(object_mask, sampling=spacing)
        min_distance_voxels = max(1, int(np.ceil(min_distance_mm / spacing.min())))
        peaks = peak_local_max(
            distance,
            min_distance=min_distance_voxels,
            threshold_abs=min_seed_radius_mm,
            labels=object_mask,
            exclude_border=False,
        )
        if not use_watershed:  # ablation : composantes connexes brutes
            labels = ndi.label(object_mask)[0].astype(np.int32)
        elif len(peaks):
            markers = np.zeros(object_mask.shape, dtype=np.int32)
            markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1, dtype=np.int32)
            labels = watershed(-distance, markers, mask=object_mask).astype(np.int32)
        else:
            labels = np.zeros(object_mask.shape, dtype=np.int32)

        voxel_volume = float(np.prod(spacing))
        voxel_length = float(np.cbrt(voxel_volume))
        candidates = []
        rejected = []
        skeleton_volume = np.zeros(labels.shape, dtype=bool) if return_intermediates else None
        for candidate_id, slices in enumerate(ndi.find_objects(labels), start=1):
            if slices is None:
                continue
            region = labels[slices] == candidate_id
            if truncate_factor > 0:
                region = _truncate_to_ball(region, distance[slices], spacing, truncate_factor)
            voxel_count = int(region.sum())
            volume_mm3 = voxel_count * voxel_volume
            diameter_mm = float(np.cbrt(6.0 * volume_mm3 / np.pi))
            if not min_diameter_mm <= diameter_mm <= max_diameter_mm:
                if return_intermediates:
                    rejected.append({"label": candidate_id, "reason": "diameter",
                                     "diameter_mm": diameter_mm})
                continue

            skeleton = skeletonize(region)
            skeleton_count = int(skeleton.sum())
            skeleton_length_mm = skeleton_count * voxel_length
            skeleton_neighbors = ndi.convolve(
                skeleton.astype(np.uint8), np.ones((3, 3, 3), dtype=np.uint8),
                mode="constant", cval=0,
            ) - skeleton
            branch_points = int(np.count_nonzero(skeleton & (skeleton_neighbors >= 3)))
            if (skeleton_length_mm / max(diameter_mm, 1e-6) > max_skeleton_ratio
                    or branch_points > max_branch_points):
                if return_intermediates:
                    rejected.append({"label": candidate_id, "reason": "skeleton",
                                     "diameter_mm": diameter_mm,
                                     "skeleton_ratio": skeleton_length_mm / max(diameter_mm, 1e-6),
                                     "branch_points": branch_points})
                continue

            local_center = np.argwhere(region).mean(axis=0)
            center = local_center + np.array([axis.start for axis in slices]) + lower
            candidate = {
                "candidate_id": len(candidates) + 1,
                "label_id": int(candidate_id),
                "z_voxel": float(center[0]),
                "y_voxel": float(center[1]),
                "x_voxel": float(center[2]),
                "z_mm_from_origin": float(center[0] * spacing[0]),
                "y_mm_from_origin": float(center[1] * spacing[1]),
                "x_mm_from_origin": float(center[2] * spacing[2]),
                "volume_mm3": float(volume_mm3),
                "diameter_mm": diameter_mm,
                "max_radius_mm": float(distance[slices][region].max()),
                "skeleton_length_mm": float(skeleton_length_mm),
                "branch_points": branch_points,
                "mean_hu": float(cropped_volume[slices][region].mean()),
            }
            candidates.append(candidate)
            if return_intermediates:
                skeleton_volume[slices] |= skeleton

    if not return_intermediates:
        return candidates
    intermediates = {
        "object_mask": object_mask,
        "watershed_labels": labels,
        "skeleton": skeleton_volume if object_mask.any() else np.zeros(labels.shape, dtype=bool),
        "crop_origin_zyx": tuple(int(value) for value in lower),
        "rejected": rejected if object_mask.any() else [],
    }
    return candidates, intermediates


def evaluate_candidates(candidates, annotations_xyz, transform_index_to_physical):
    """Calcule le rappel et les faux positifs avec un appariement un-à-un.

    `annotations_xyz` est un tableau (n, 4): coordonnées monde x/y/z en mm
    et diamètre annoté en mm. `transform_index_to_physical` convertit un index
    continu (x, y, z) en coordonnées monde, par exemple via SimpleITK.
    """
    annotations = np.asarray(annotations_xyz, dtype=float)
    if annotations.size == 0:
        annotations = np.empty((0, 4), dtype=float)
    if annotations.ndim != 2 or annotations.shape[1] != 4:
        raise ValueError("annotations_xyz doit avoir la forme (n, 4)")

    candidate_world = []
    for candidate in candidates:
        index_xyz = (
            candidate["x_voxel"], candidate["y_voxel"], candidate["z_voxel"]
        )
        candidate_world.append(transform_index_to_physical(index_xyz))
    candidate_world = np.asarray(candidate_world, dtype=float).reshape(-1, 3)

    matched_candidate_ids = set()
    matched_annotation_ids = set()
    if len(candidate_world) and len(annotations):
        distances = np.linalg.norm(
            candidate_world[:, None, :] - annotations[None, :, :3], axis=2
        )
        radii = annotations[:, 3] / 2.0
        costs = np.where(distances <= radii[None, :], distances, 1e9)
        candidate_ids, annotation_ids = linear_sum_assignment(costs)
        for candidate_id, annotation_id in zip(candidate_ids, annotation_ids):
            if costs[candidate_id, annotation_id] < 1e9:
                matched_candidate_ids.add(int(candidate_id))
                matched_annotation_ids.add(int(annotation_id))

    n_annotations = len(annotations)
    n_matched = len(matched_annotation_ids)
    false_positives = len(candidates) - len(matched_candidate_ids)
    return {
        "nodules": n_annotations,
        "nodules_retrouves": n_matched,
        "rappel": n_matched / n_annotations if n_annotations else np.nan,
        "candidats": len(candidates),
        "faux_positifs": false_positives,
        "faux_positifs_par_scan": false_positives,
    }