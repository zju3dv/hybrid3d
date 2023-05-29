import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm


def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)


def plot_pc(pc_np, z_cutoff=1000, birds_view=False, color='height', size=0.3, ax=None, cmap=cm.viridis, vmin=None, vmax=None, alpha=None, is_equal_axes=True):
    # remove large z points
    valid_index = pc_np[:, 0] < z_cutoff
    pc_np = pc_np[valid_index, :]

    if ax is None:
        fig = plt.figure(figsize=(9, 9))
        ax = Axes3D(fig)
    if type(color) == str and color == 'height':
        c = pc_np[:, 2]
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=size, c=c, cmap=cmap, edgecolors=None)
    elif type(color) == str and color == 'reflectance':
        assert False
    elif type(color) == np.ndarray:
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=size, c=color, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, edgecolors=None)
    else:
        ax.scatter(pc_np[:, 0], pc_np[:, 1], pc_np[:, 2], s=size, c=color, edgecolors=None)

    if is_equal_axes:
        axisEqual3D(ax)
    if True == birds_view:
        ax.view_init(elev=0, azim=-90)
    else:
        ax.view_init(elev=-45, azim=-90)
    # ax.invert_yaxis()

    return ax
