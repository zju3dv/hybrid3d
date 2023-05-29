import numpy as np
import cv2

def dense_to_sparse(depth, num_samples, max_depth=20.0):
    """
    Samples pixels with `num_samples`/#pixels probability in `depth`.
    Only pixels with a maximum depth of `max_depth` are considered.
    If no `max_depth` is given, samples in all pixels
    """
    mask_keep = depth > 0
    if max_depth is not np.inf:
        mask_keep = np.bitwise_and(mask_keep, depth <= max_depth)
    n_keep = np.count_nonzero(mask_keep)
    if n_keep == 0:
        return mask_keep * depth
    else:
        prob = float(num_samples) / n_keep
        return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob) * depth
