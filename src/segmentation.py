import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.morphology import ball, disk
from skimage.segmentation import clear_border


def resample_isotropic(img, spacing=1.0):
    """Rééchantillonne en voxels cubiques (spacing en mm)."""
    old_size = np.array(img.GetSize())
    old_spacing = np.array(img.GetSpacing())
    new_size = np.round(old_size * old_spacing / spacing).astype(int)
    rs = sitk.ResampleImageFilter()
    rs.SetOutputSpacing((spacing,) * 3)
    rs.SetSize([int(s) for s in new_size])
    rs.SetOutputOrigin(img.GetOrigin())
    rs.SetOutputDirection(img.GetDirection())
    rs.SetInterpolator(sitk.sitkLinear)
    rs.SetDefaultPixelValue(-2000)
    return rs.Execute(img)


def segment_lungs(vol, thr=-320, r_open=2, r_close=8):
    """vol : tableau (z, y, x) en HU. Retourne un masque booléen des poumons."""
    binary = vol < thr
    for k in range(binary.shape[0]):                 # air extérieur
        binary[k] = clear_border(binary[k])
    binary = ndi.binary_opening(binary, structure=ball(r_open))

    lab, n = ndi.label(binary)
    if n == 0:
        return binary
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = [i for i in np.argsort(counts)[-2:]
            if counts[i] >= 0.1 * counts.max()]      # les 2 plus gros
    mask = np.isin(lab, keep)

    se = disk(r_close)                               # fermeture 2D par coupe
    for k in range(mask.shape[0]):
        if mask[k].any():
            mask[k] = ndi.binary_fill_holes(ndi.binary_closing(mask[k], structure=se))
    return mask


def nodule_voxels(img, ann_rows):
    """Coordonnées monde (mm) des nodules -> indices (z, y, x)."""
    sp = np.array(img.GetSpacing())
    org = np.array(img.GetOrigin())
    c = (ann_rows[["coordX", "coordY", "coordZ"]].values - org) / sp
    return np.rint(c).astype(int)[:, ::-1]


def nodules_inside(mask, voxels):
    """Nombre de nodules dont le centre est dans le masque."""
    ok = 0
    for z, y, x in voxels:
        if (0 <= z < mask.shape[0] and 0 <= y < mask.shape[1]
                and 0 <= x < mask.shape[2] and mask[z, y, x]):
            ok += 1
    return ok