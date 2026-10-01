# hsi_preprocessing.py

import numpy as np
import h5py
import scipy.io as sio
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------
# 1. Universal MAT-file loader
# ---------------------------------------------------
def load_mat_variable(mat_path, key):
    if h5py.is_hdf5(mat_path):
        with h5py.File(mat_path, 'r') as f:
            if key not in f:
                raise KeyError(f"{key} not found in {mat_path}")
            data = np.array(f[key])
    else:
        mat = sio.loadmat(mat_path)
        if key not in mat:
            raise KeyError(f"{key} not found in {mat_path}")
        data = mat[key]
    return data


# ---------------------------------------------------
# 2. Load HSI + GT
# ---------------------------------------------------
def load_hsi_data(hsi_mat_path, gt_mat_path,
                  hsi_key="ori_data", gt_key="map"):

    hsi = load_mat_variable(hsi_mat_path, hsi_key)
    gt = load_mat_variable(gt_mat_path, gt_key)

    if gt.ndim != 2:
        raise ValueError("GT must be 2D")

    H_gt, W_gt = gt.shape
    dims = hsi.shape

    if dims[1] == H_gt and dims[2] == W_gt:
        hsi = np.transpose(hsi, (1, 2, 0))
    elif dims[0] == H_gt and dims[2] == W_gt:
        hsi = np.transpose(hsi, (0, 2, 1))
    elif dims[0] == H_gt and dims[1] == W_gt:
        pass
    elif dims[0] == W_gt and dims[1] == H_gt:
        hsi = np.transpose(hsi, (1, 0, 2))
    else:
        raise ValueError("Cannot infer HSI layout")

    # NaN handling
    nan_mask = np.isnan(hsi).any(axis=-1)
    hsi[nan_mask] = 0.0
    gt[nan_mask] = 0

    return hsi.astype(np.float32), gt.astype(np.int64)


# ---------------------------------------------------
# 3. Full Spectral Normalization
# ---------------------------------------------------
def normalize_hsi(hsi,
                  use_global_max=True,
                  use_band_standardization=True,
                  use_l2_norm=True):

    H, W, B = hsi.shape
    reshaped = hsi.reshape(-1, B)

    # 1️⃣ Global max scaling
    if use_global_max:
        max_val = reshaped.max()
        if max_val > 0:
            reshaped = reshaped / max_val

    # 2️⃣ Band-wise standardization
    if use_band_standardization:
        scaler = StandardScaler()
        reshaped = scaler.fit_transform(reshaped)

    # 3️⃣ Spectral L2 normalization
    if use_l2_norm:
        norms = np.linalg.norm(reshaped, axis=1, keepdims=True)
        norms[norms == 0] = 1
        reshaped = reshaped / norms

    return reshaped.reshape(H, W, B)


# ---------------------------------------------------
# 4. Zero Padding
# ---------------------------------------------------
def zero_pad_hsi(hsi, margin):
    return np.pad(
        hsi,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="constant",
        constant_values=0
    )


# ---------------------------------------------------
# 5. Patch Extraction with Ignored Labels
# ---------------------------------------------------
def extract_patches_stride(
    hsi,
    gt,
    patch_size=13,
    stride=1,
    ignored_labels=None
):

    assert patch_size % 2 == 1

    margin = patch_size // 2
    H, W, B = hsi.shape
    hsi_padded = zero_pad_hsi(hsi, margin)

    # Ensure ignored_labels is iterable (convert int to list/set)
    if ignored_labels is not None and isinstance(ignored_labels, (int, np.integer)):
        ignored_labels = [ignored_labels]

    patches, labels, locations = [], [], []

    for i in range(0, H, stride):
        for j in range(0, W, stride):

            label = gt[i, j]

            if ignored_labels is not None:
                if label in ignored_labels:
                    continue
            elif label == 0:
                continue

            patch = hsi_padded[i:i+patch_size,
                               j:j+patch_size, :]

            if patch.shape[:2] != (patch_size, patch_size):
                continue

            patches.append(patch)
            labels.append(label)
            locations.append((i, j))

    return (
        np.asarray(patches, dtype=np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(locations, dtype=np.int32),
    )


# # ---------------------------------------------------
# # 6. Full Pipeline
# # ---------------------------------------------------
def preprocess_hsi_dataset(
    hsi_mat_path,
    gt_mat_path,
    patch_size=13,
    stride=1,
    ignored_labels=None
):

    hsi, gt = load_hsi_data(hsi_mat_path, gt_mat_path)
    hsi = normalize_hsi(hsi)

    patches, labels, locations = extract_patches_stride(
        hsi,
        gt,
        patch_size=patch_size,
        stride=stride,
        ignored_labels=ignored_labels
    )

    print("HSI preprocessing completed")
    print("HSI shape:", hsi.shape)
    print("Patches:", patches.shape)

    return patches, labels, locations

