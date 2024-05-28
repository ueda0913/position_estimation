# 12個のノードの精度の平均値、最後に最大最小のノードの推移
import os
import pickle

import matplotlib.pyplot as plt

project_path = "../../data-raid/static/WAFL_pos_estimation"
dir_name = "vit_wafl_raw_iid"
image_file_name = "node_avg"
image_path = os.path.join(project_path, dir_name, "images")  # 出力先のFolder

# 各エポックの精度を記録した配列をloadする。
with open(
    os.path.join(project_path, dir_name, "params", "historys_data.pkl"), "rb"
) as f:
    his = pickle.load(f)

# ノードの平均を求める
# vit_b16
n_node = len(his)
num_epoch = len(his[0])
vit_avg = []
min_index = 0
max_index = 0
for epoch in range(0, num_epoch):
    avg = 0
    for node in range(0, n_node):
        avg += his[node][epoch][4]
    avg = avg / n_node
    vit_avg.append(avg)

for node in range(0, n_node):
    if his[node][-1][4] > his[max_index][-1][4]:
        max_index = node
    if his[node][-1][4] < his[min_index][-1][4]:
        min_index = node
print(f"max:{max_index}, min:{min_index}")
# plotする
epoch_array = [i for i in range(0, num_epoch)]
plt.plot(epoch_array, vit_avg, label="Average")
plt.plot(
    epoch_array,
    his[min_index][:, 4],
    label=f"Minimum node (node{min_index})",
    linewidth=0.75,
)
plt.plot(
    epoch_array,
    his[max_index][:, 4],
    label=f"Maximum node (node{max_index})",
    linewidth=0.75,
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xlim([-100, 3100])
plt.ylim([0.1, 0.9])
plt.legend()
# plt.savefig(os.path.join(image_path, f"{image_file_name}_acc.png"), bbox_inches="tight")
