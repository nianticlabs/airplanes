import numpy as np


def compute_symmetric_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute intersection over union given two masks.
    Args:
        pred: binary mask, (H,W)
        gt: binary mask, (H,W)
    Returns:
        IoU
    """
    return 0.5 * (compute_iou(pred=pred, gt=gt) + compute_iou(pred=1 - pred, gt=1 - gt))


def compute_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute intersection over union given two masks.
    Args:
        pred: binary mask; floating point values will be treated as True if nonzero. (B,H,W)
        gt: binary mask; floating point values will be treated as True if nonzero. (B,H,W)
    Returns:
        IoU
    """
    intersection = np.logical_and(pred, gt).mean()  # type: ignore
    union = np.logical_or(pred, gt).mean()  # type: ignore
    return intersection / union
