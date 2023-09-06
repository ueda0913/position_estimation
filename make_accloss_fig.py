import os
import pickle

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np

data_dir = "../data-raid/data/position_estimation_dataset"
project_path = "../data-raid/static/WAFL_pos_estimation"
contact_pattern_dir = "../data-raid/static/contact_pattern"
classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
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
    plt.plot(historys[i][:, 0], historys[i][:, 1], label=f"node{i}(training)")
    plt.plot(historys[i][:, 0], historys[i][:, 3], label=f"node{i}(validation)")
plt.xticks(np.arange(0, num_epochs + 1, unit))
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Learning curve (loss)")
plt.legend(ncol=2)
plt.xlim([0, 500])
plt.savefig(os.path.join(cur_dir, f"images/{filename}_loss.png"))

plt.figure(figsize=(9, 8))
for i in range(len(historys)):
    plt.plot(historys[i][:, 0], historys[i][:, 2], label=f"node{i}(training)")
    plt.plot(historys[i][:, 0], historys[i][:, 4], label=f"node{i}(validation)")
plt.xticks(np.arange(0, num_epochs + 1, unit))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Learning curve (accuracy)")
plt.legend(ncol=2)
plt.xlim([0, 500])
plt.savefig(os.path.join(cur_dir, f"images/{filename}_acc.png"))
