import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# plt.rcParams["font.family"] = "Times New Roman"

font_size = 12
font_size_legend = 8

file_ablation = str(Path.home()) + '/Dropbox/hybrid_feature_assets/repeatability/extend_expr.txt'
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
    # ['max-view-pooling-0.05', 'Max-View Pooling (0.05)', 'tab:blue'],
    # ['max-view-pooling-0.075', 'Max-View Pooling (0.075)', 'tab:green'],
    # ['max-view-pooling-0.1', 'Max-View Pooling (0.1)', 'tab:purple'],
    # ['soft-view-pooling-0.05', 'Soft-View Pooling (0.005)', 'tab:brown'],
    # ['soft-view-pooling-0.075', 'Soft-View Pooling (0.075)', 'tab:pink'],
    # ['soft-view-pooling-0.1', 'Soft-View Pooling (0.1)', 'k'],
    # ['fuception-0.005', 'Fuception (0.005)', 'tab:olive'],
    # ['fuception-0.075', 'Fuception (0.075)', 'tab:cyan'],
    # ['fuception-0.1', 'Fuception (0.1)', 'goldenrod'],
    ['d3feat-score-loss', 'D3Feat Score Loss', 'b'],
    ['d3feat-score-full', 'D3Feat Score Strategy', 'm'],
    ['Random', 'Random', 'tab:gray'],

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
plt.savefig('repeatability_supp_d3feat.pdf', format='pdf')
plt.show()
