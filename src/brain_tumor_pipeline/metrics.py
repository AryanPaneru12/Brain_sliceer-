from __future__ import annotations

import numpy as np
from scipy import ndimage
from tensorflow.keras import backend as K


def tversky(y_true, y_pred, smooth: float = 1e-6):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1.0 - y_pred_pos))
    false_pos = K.sum((1.0 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (
        true_pos + alpha * false_neg + (1.0 - alpha) * false_pos + smooth
    )


def tversky_loss(y_true, y_pred):
    return 1.0 - tversky(y_true, y_pred)


def focal_tversky(y_true, y_pred):
    gamma = 0.75
    return K.pow(1.0 - tversky(y_true, y_pred), gamma)


def dice_coefficient(y_true, y_pred, smooth: float = 1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)


def iou_coefficient(y_true, y_pred, smooth: float = 1e-6):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


def dice_numpy(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    true = np.asarray(y_true) > 0
    pred = np.asarray(y_pred) >= threshold
    denom = true.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * np.logical_and(true, pred).sum() / denom)


def _surface_distances(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    true = np.asarray(y_true).astype(bool)
    pred = np.asarray(y_pred).astype(bool)
    if true.sum() == 0 and pred.sum() == 0:
        return np.array([0.0], dtype=np.float64)
    if true.sum() == 0 or pred.sum() == 0:
        return np.array([np.inf], dtype=np.float64)

    footprint = ndimage.generate_binary_structure(true.ndim, 1)
    true_surface = true ^ ndimage.binary_erosion(true, structure=footprint, border_value=0)
    pred_surface = pred ^ ndimage.binary_erosion(pred, structure=footprint, border_value=0)
    true_to_pred = ndimage.distance_transform_edt(~pred_surface)[true_surface]
    pred_to_true = ndimage.distance_transform_edt(~true_surface)[pred_surface]
    return np.concatenate([true_to_pred, pred_to_true]).astype(np.float64)


def hausdorff_distance(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    distances = _surface_distances(np.asarray(y_true) > 0, np.asarray(y_pred) >= threshold)
    return float(np.max(distances))


def hausdorff95(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    distances = _surface_distances(np.asarray(y_true) > 0, np.asarray(y_pred) >= threshold)
    return float(np.percentile(distances, 95))


CUSTOM_OBJECTS = {
    "tversky": tversky,
    "tversky_loss": tversky_loss,
    "focal_tversky": focal_tversky,
    "dice_coefficient": dice_coefficient,
    "iou_coefficient": iou_coefficient,
}
