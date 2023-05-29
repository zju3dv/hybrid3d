import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# plt.rcParams["font.family"] = "Times New Roman"

font_size = 12
font_size_legend = 8

file_ablation = str(Path.home()) + '/Dropbox/hybrid_feature_assets/repeatability/ablation.txt'
data_ablation = pd.read_csv(file_ablation, sep="\t", index_col=0)

print(type(data_ablation))
print(data_ablation)

def get_val(data, key):
    return data[key].values

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
fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(111)

for i, info in enumerate(y_infos):
    key, name, color = info
    ax.plot(np.arange(len(xs)), get_val(data_ablation, key) * 100, label=name, color=color)

# ax.set_title('Test')
ax.set_xlabel('Keypoint Numbers', fontsize=font_size)
ax.set_ylabel('Repeatability %', fontsize=font_size)
ax.set_xticks(np.arange(len(xs)))
ax.set_xticklabels(xs)
ax.set_xlim(0, len(xs)-1)
ax.legend(fontsize=font_size_legend)
ax.grid(color=(0.7, 0.7, 0.7), linestyle='--')
plt.tight_layout()
plt.savefig('repeatability_ablation.pdf', format='pdf')
plt.show()
