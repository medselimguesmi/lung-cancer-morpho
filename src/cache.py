"""Chargement d'un scan rééchantillonné et segmenté, avec cache disque (data/cache)."""
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.segmentation import resample_isotropic, segment_lungs  # noqa: E402

CACHE_DIR = ROOT / "data" / "cache"


def find_scan(scan_dir, seriesuid):
    matches = list(Path(scan_dir).rglob(f"{seriesuid}.mhd"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Scan attendu une fois, trouvé {len(matches)} : {seriesuid}")
    return matches[0]


def _geometry_image(origin, spacing, direction):
    """Image vide portant seulement la géométrie, pour convertir index -> mm."""
    geometry = sitk.Image(1, 1, 1, sitk.sitkInt8)
    geometry.SetOrigin(tuple(origin))
    geometry.SetSpacing(tuple(spacing))
    geometry.SetDirection(tuple(direction))
    return geometry


def load_scan(patient, config, luna_dir):
    """Retourne (volume HU int16 (z, y, x), masque pulmonaire, géométrie sitk)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = (f"{patient.seriesuid}_{config['resample_mm']}_{config['thr']}_"
           f"{config['r_open']}_{config['r_close']}.npz")
    path = CACHE_DIR / key
    if path.exists():
        data = np.load(path)
        return data["volume"], data["mask"], _geometry_image(
            data["origin"], data["spacing"], data["direction"])
    image = sitk.ReadImage(str(find_scan(Path(luna_dir) / patient.subset, patient.seriesuid)))
    resampled = resample_isotropic(image, config["resample_mm"])
    volume = sitk.GetArrayFromImage(resampled)
    mask = segment_lungs(volume, thr=config["thr"], r_open=config["r_open"],
                         r_close=config["r_close"])
    volume = np.clip(volume, -2000, 3000).astype(np.int16)
    np.savez(path, volume=volume, mask=mask, origin=resampled.GetOrigin(),
             spacing=resampled.GetSpacing(), direction=resampled.GetDirection())
    return volume, mask, _geometry_image(
        resampled.GetOrigin(), resampled.GetSpacing(), resampled.GetDirection())
