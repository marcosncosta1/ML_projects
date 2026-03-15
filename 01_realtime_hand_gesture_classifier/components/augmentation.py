import numpy as np


def augment_batch(X: np.ndarray, y: np.ndarray, n_copies: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Augment a landmark dataset.

    Parameters
    ----------
    X : (N, 63) float32 — normalised landmark vectors
    y : (N,) int        — class labels
    n_copies : int      — number of augmented copies per original sample

    Returns
    -------
    X_aug, y_aug — original rows prepended to augmented rows
    """
    aug_X = [X]
    aug_y = [y]
    for _ in range(n_copies):
        aug_X.append(_augment_array(X))
        aug_y.append(y)
    return np.vstack(aug_X), np.concatenate(aug_y)


def _augment_array(X: np.ndarray) -> np.ndarray:
    """Apply randomised transforms to every row in X independently."""
    N = len(X)
    pts = X.reshape(N, 21, 3).copy()

    # 1. Gaussian noise
    pts += np.random.normal(0, 0.02, pts.shape)

    # 2. Random scale
    scale = np.random.uniform(0.85, 1.15, (N, 1, 1))
    pts *= scale

    # 3. Random rotation in the xy-plane (hand tilt)
    theta = np.random.uniform(-25, 25, N) * np.pi / 180
    c, s = np.cos(theta), np.sin(theta)
    # Build (N, 2, 2) rotation matrices
    R = np.stack([np.stack([c, -s], axis=1),
                  np.stack([s,  c], axis=1)], axis=1)   # (N, 2, 2)
    pts[:, :, :2] = np.einsum("nij,nkj->nki", R, pts[:, :, :2])

    # 4. Random translation (x, y only — normalised coords stay centred)
    shift = np.random.uniform(-0.12, 0.12, (N, 1, 2))
    pts[:, :, :2] += shift

    return pts.reshape(N, 63).astype(np.float32)
