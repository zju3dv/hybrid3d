import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# plt.rcParams["font.family"] = "Times New Roman"

font_size = 14
font_size_legend = 12

file_tau1 = str(Path.home()) + '/Dropbox/HybridFeature/hybrid_feature_assets/rebuttal_table/varying_tau_1.txt'
data_tau1 = pd.read_csv(file_tau1, sep="\t", index_col=0)

print(type(data_tau1))
print(data_tau1)

file_redwood = str(Path.home()) + '/Dropbox/HybridFeature/hybrid_feature_assets/rebuttal_table/varying_tau_2.txt'
data_tau2 = pd.read_csv(file_redwood, sep="\t", index_col=0)

print(type(data_tau2))
print(data_tau2)

def get_val(data, key):
    return data[key].values

# xs = get_val(data, 'KeypointNum')
# xs = [4, 8, 16, 32, 64, 128, 256, 512]
xs = list(np.arange(0, 0.21, 0.01))
xs_tick = list(np.arange(0, 0.21, 0.025))

print(xs)

y_infos = [
    ['SuperPoint', 'SuperPoint', 'tab:blue'],
    ['KP2D', 'KP2D', 'tab:olive'],
    ['PerfectMatch', 'PerfectMatch', 'tab:purple'],
    ['D3Feat', 'D3Feat', 'tab:green'],
    ['FCGF', 'FCGF', 'tab:gray'],
    ['Our_2D_RGB', 'Our 2D (RGB)', 'tab:cyan'],
    ['Our_3D', 'Our 3D', 'tab:red'],
]

# fig, ax = plt.subplots(221)
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121)

for i, info in enumerate(y_infos):
    key, name, color = info
    ax.plot(xs, get_val(data_tau1, key), label=name, color=color)

# ax.set_title('3DMatch', fontsize=font_size)
ax.set_xlabel('Inlier Ratio Threshold', fontsize=font_size)
ax.set_ylabel('Feature Matching Recall %', fontsize=font_size)
# ax.set_xticks(np.arange(len(xs)))
ax.set_xticks(xs_tick)
# ax.set_xticklabels(xs_tick)
ax.set_xlim(xs_tick[0], xs_tick[-1])
ax.set_ylim(0, 100)
ax.legend(fontsize=font_size_legend)
# ax.grid(color=(0.7, 0.7, 0.7), linestyle='--')
plt.axvline(x=0.05, color=(0.7, 0.7, 0.7), linestyle='--', linewidth=1)
# plt.show()

ax = fig.add_subplot(122)

for i, info in enumerate(y_infos):
    key, name, color = info
    ax.plot(xs, get_val(data_tau2, key), label=name, color=color)

# ax.set_title('Redwood', fontsize=font_size)
ax.set_xlabel('Inlier Distance Threshold (m)', fontsize=font_size)
ax.set_ylabel('Feature Matching Recall %', fontsize=font_size)
# ax.set_xticks(np.arange(len(xs)))
# ax.set_xticklabels(xs)
# ax.set_xlim(0, len(xs)-1)
ax.set_xticks(xs_tick)
ax.set_xlim(xs_tick[0], xs_tick[-1])
ax.set_ylim(0, 100)
ax.legend(fontsize=font_size_legend)
plt.axvline(x=0.1, color=(0.7, 0.7, 0.7), linestyle='--', linewidth=1)
# ax.grid(color=(0.7, 0.7, 0.7), linestyle='--')
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='center right')
plt.tight_layout()
# fig.subplots_adjust(right=0.84)
plt.savefig('varying_tau.pdf', format='pdf')
plt.show()

