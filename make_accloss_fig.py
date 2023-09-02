import os
import pickle

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

data_dir = "../data-raid/data/UTokyoE_building_dataset"
project_path = "../data-raid/static/WAFL_research"
noniid_filter_dir = "../data-raid/static/WAFL_research/noniid_filter"
contact_pattern_dir = "../data-raid/static/contact_pattern"
classes = ("安田講堂", "工2", "工3", "工13", "工4", "工8", "工1", "工6", "列品館", "法文1")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")

# change area
load_time_index = "2023-06-26-13"
filename = "pre-train-only"


cur_dir = os.path.join(project_path, load_time_index)
with open(
    os.path.join(project_path, load_time_index, "params", "historys_data.pkl"),
    "rb",
) as f:
    historys = pickle.load(f)

num_epochs = len(historys[0])
unit = num_epochs / 10

img_dir_path = os.path.join(cur_dir, "images")
if not (os.path.exists(img_dir_path)) or os.path.isfile(img_dir_path):
    os.makedirs(img_dir_path)

plt.figure(figsize=(9, 8))
for i in range(len(historys)):
    plt.plot(historys[i][:, 0], historys[i][:, 1], label=f"node{i}(訓練)")
    plt.plot(historys[i][:, 0], historys[i][:, 3], label=f"node{i}(検証)")
plt.xticks(np.arange(0, num_epochs + 1, unit))
plt.xlabel("繰り返し回数")
plt.ylabel("損失")
plt.title("学習曲線(損失)")
plt.legend(ncol=2)
plt.xlim([0, 500])
plt.savefig(os.path.join(cur_dir, f"images/{filename}_loss.png"))

plt.figure(figsize=(9, 8))
for i in range(len(historys)):
    plt.plot(historys[i][:, 0], historys[i][:, 2], label=f"node{i}(訓練)")
    plt.plot(historys[i][:, 0], historys[i][:, 4], label=f"node{i}(検証)")
plt.xticks(np.arange(0, num_epochs + 1, unit))
plt.xlabel("繰り返し回数")
plt.ylabel("精度")
plt.title("学習曲線(精度)")
plt.legend(ncol=2)
plt.xlim([0, 500])
plt.savefig(os.path.join(cur_dir, f"images/{filename}_acc.png"))
