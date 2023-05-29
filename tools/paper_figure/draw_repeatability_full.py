import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# plt.rcParams["font.family"] = "Times New Roman"

font_size = 16
font_size_legend = 10

file_3dm = str(Path.home()) + '/Dropbox/hybrid_feature_assets/repeatability/3dmatch.txt'
data_3dm = pd.read_csv(file_3dm, sep="\t", index_col=0)

print(type(data_3dm))
print(data_3dm)

file_redwood = str(Path.home()) + '/Dropbox/hybrid_feature_assets/repeatability/redwood.txt'
data_redwood = pd.read_csv(file_redwood, sep="\t", index_col=0)

print(type(data_redwood))
print(data_redwood)

def get_val(data, key):
    return data[key].values

# xs = get_val(data, 'KeypointNum')
xs = [4, 8, 16, 32, 64, 128, 256, 512]

print(xs)

y_infos = [
    ['SuperPoint', 'SuperPoint', 'tab:blue'],
    ['KP2D', 'KP2D', 'tab:olive'],
    # ['USIP_nms_0.03', 'USIP', 'tab:cyan'],
    ['USIP', 'USIP', 'tab:purple'],
    ['D3Feat', 'D3Feat', 'tab:green'],
    ['Our_2D_rgb', 'Our 2D (RGB)', 'tab:cyan'],
    # ['Our_3D', 'Our 3D', 'tab:orange'],
    ['Our_3D', 'Our 3D', 'tab:red'],
    ['Random', 'Random', 'tab:gray'],

]

# fig, ax = plt.subplots(221)
fig = plt.figure(figsize=(17, 4))
ax = fig.add_subplot(131)

for i, info in enumerate(y_infos):
    key, name, color = info
    ax.plot(np.arange(len(xs)), get_val(data_3dm, key) * 100, label=name, color=color)

ax.set_title('3DMatch', fontsize=font_size)
ax.set_xlabel('Keypoint Numbers', fontsize=font_size)
ax.set_ylabel('Repeatability %', fontsize=font_size)
ax.set_xticks(np.arange(len(xs)))
ax.set_xticklabels(xs)
ax.set_xlim(0, len(xs)-1)
ax.legend(fontsize=font_size_legend)
ax.grid(color=(0.7, 0.7, 0.7), linestyle='--')
# plt.show()

ax = fig.add_subplot(132)

for i, info in enumerate(y_infos):
    key, name, color = info
    ax.plot(np.arange(len(xs)), get_val(data_redwood, key) * 100, label=name, color=color)

ax.set_title('Redwood', fontsize=font_size)
ax.set_xlabel('Keypoint Numbers', fontsize=font_size)
ax.set_ylabel('Repeatability %', fontsize=font_size)
ax.set_xticks(np.arange(len(xs)))
ax.set_xticklabels(xs)
ax.set_xlim(0, len(xs)-1)
ax.legend(fontsize=font_size_legend)
ax.grid(color=(0.7, 0.7, 0.7), linestyle='--')
plt.tight_layout()
# plt.savefig('repeatability.pdf', format='pdf')
# plt.show()



file_ablation = str(Path.home()) + '/Dropbox/hybrid_feature_assets/repeatability/ablation.txt'
data_ablation = pd.read_csv(file_ablation, sep="\t", index_col=0)

# xs = get_val(data, 'KeypointNum')
xs = [4, 8, 16, 32, 64, 128, 256, 512]

print(xs)

y_infos = [
    ['3D_final', 'Our 3D', 'tab:red'],
    ['2D_rgb_only', 'Our 2D (RGB)', 'tab:blue'],
    ['2D_fuse', 'Our 2D (fuse)', 'tab:green'],
    # ['3D_rgb_only', '3D (w/o) 2D fuse', 'tab:green'],
    ['3d_no_balance', 'Our 3D (w/o balance.)', 'tab:purple'],
    ['3d_no_consis', 'Our 3D (w/o cons.)', 'tab:brown'],
    ['3d_no_peak_loss', 'Our 3D (w/o peak.)', 'tab:pink'],
    ['fcgf_our_score', 'FCGF (score)', 'k'],
    ['sparse_1k_fcgf_score', 'Sparse 5k: FCGF (score)', 'tab:olive'],
    ['sparse_1k_our', 'Sparse 5k: Our', 'tab:cyan'],
    ['sparse_5k_fcgf_score', 'Sparse 1k: FCGF (score)', 'tab:gray'],
    ['sparse_5k_our', 'Sparse 1k: Our', 'b'],

]

# fig, ax = plt.subplots(221)
# fig = plt.figure(figsize=(5, 4))
# ax = fig.add_subplot(111)
ax = fig.add_subplot(133)

for i, info in enumerate(y_infos):
    key, name, color = info
    ax.plot(np.arange(len(xs)), get_val(data_ablation, key) * 100, label=name, color=color)

ax.set_title('Ablation on 3DMatch', fontsize=font_size)
ax.set_xlabel('Keypoint Numbers', fontsize=font_size)
ax.set_ylabel('Repeatability %', fontsize=font_size)
ax.set_xticks(np.arange(len(xs)))
ax.set_xticklabels(xs)
ax.set_xlim(0, len(xs)-1)
ax.legend(fontsize=font_size_legend)
ax.grid(color=(0.7, 0.7, 0.7), linestyle='--')
plt.tight_layout()
plt.savefig('repeatability_full.pdf', format='pdf')
plt.show()
