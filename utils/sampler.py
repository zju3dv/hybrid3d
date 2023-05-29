import numpy as np


class FarthestSampler:
    def __init__(self):
        pass

    def calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def sample(self, pts, k):
        farthest_pts = np.zeros((k, 3))
        if pts.size == 0:
            return np.empty((0, 3))
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self.calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum(
                distances, self.calc_distances(farthest_pts[i], pts))
        return farthest_pts


def nms_3D(keypoints_np, sigmas_np, NMS_radius, max_selected_pts=99999999, method='max'):
    '''
    3D Non-Max-Suppression
    :param keypoints_np: Mx3
    :param sigmas_np: M
    :return: valid_keypoints_np, valid_sigmas_np
    '''
    indices = np.arange(0, keypoints_np.shape[0])
    if NMS_radius <= 0 or keypoints_np.shape[0] <=1:
        return keypoints_np, sigmas_np, indices
    if sigmas_np is None:
        sigmas_np = np.ones((keypoints_np.shape[0]))
    valid_keypoint_counter = 0
    valid_keypoints_np = np.empty(keypoints_np.shape, dtype=keypoints_np.dtype)
    valid_sigmas_np = np.empty(sigmas_np.shape, dtype=sigmas_np.dtype)
    valid_indices_np = np.empty(indices.shape, dtype=indices.dtype)

    while keypoints_np.shape[0] > 0:
        if method == 'max':
            current_idx = np.argmax(sigmas_np, axis=0)
        elif method == 'min':
            current_idx = np.argmin(sigmas_np, axis=0)
        else:
            raise ValueError('Unkonwn method : %s' % method)

        valid_keypoints_np[valid_keypoint_counter, :] = keypoints_np[current_idx, :]
        valid_sigmas_np[valid_keypoint_counter] = sigmas_np[current_idx]
        valid_indices_np[valid_keypoint_counter] = indices[current_idx]
        # remove the rows that within a certain radius of the selected minimum
        distance_array = np.linalg.norm((keypoints_np[current_idx, :] - keypoints_np), axis=1, keepdims=False)  # M
        mask = distance_array > NMS_radius  # M

        keypoints_np = keypoints_np[mask, ...]
        sigmas_np = sigmas_np[mask]
        indices = indices[mask]

        # increase counter
        valid_keypoint_counter += 1
        # stop when reached max_count
        if valid_keypoint_counter >= max_selected_pts:
            break

    return valid_keypoints_np[:valid_keypoint_counter, :], valid_sigmas_np[:valid_keypoint_counter], valid_indices_np[:valid_keypoint_counter]
