"""Initial rigid XY alignment shared by inference export and evaluation."""
import numpy as np


def initial_transform(pred, ref, i, j):
    """Anchor at i and align i-to-j displacement directions; no scale.

    Positions are (N, 2) metres. Return a local-to-reference rotation,
    translation in metres, and counterclockwise rotation angle in radians.
    """
    x, y = pred[j] - pred[i], ref[j] - ref[i]
    lengths = np.array([np.linalg.norm(x), np.linalg.norm(y)])
    if not np.isfinite(lengths).all() or np.any(lengths == 0):
        raise ValueError("Initial direction is undefined: zero or nonfinite displacement")
    angle = np.arctan2(x[0] * y[1] - x[1] * y[0], np.dot(x, y))
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    translation = ref[i] - pred[i] @ rotation.T
    return rotation, translation, float(angle)
