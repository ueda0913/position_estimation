import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

project_path = "../../data-raid/static/WAFL_pos_estimation"

# change area
load_dir_name = "2023-11-02-00"
filename = "validation-only"


cur_dir = os.path.join(project_path, load_dir_name)
with open(
    os.path.join(project_path, load_dir_name, "params", "historys_data.pkl"),
    "rb",
) as f:
    historys = pickle.load(f)

# numpy_array = np.linspace(0, 0.9, len(historys) * 2)
# color_list = [(x, x, x) for x in numpy_array]
color_list = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#b5bdab",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#adadad",
    "#ff7f0e",
    "#1f77b4",
    "#8c564b",
]
img_dir_path = os.path.join(cur_dir, "images")
if not (os.path.exists(img_dir_path)) or os.path.isfile(img_dir_path):
    os.makedirs(img_dir_path)
num_epochs = len(historys[0])
unit = num_epochs / 10

plt.figure(figsize=(9, 8))
for i in range(len(historys)):
    # plt.plot(
    #     historys[i][:, 0],
    #     historys[i][:, 2],
    #     label=f"node{i}(training)",
    #     linewidth=0.5,
    #     color=color_list[i * 2],
    # )
    plt.plot(
        historys[i][:, 0],
        historys[i][:, 4],
        label=f"node{i}(validation)",
        linewidth=0.5,
        color=color_list[i * 2 + 1],
    )
plt.xticks(np.arange(0, num_epochs + 1, unit))
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Learning curve (accuracy)")
plt.legend(ncol=2)
plt.savefig(os.path.join(cur_dir, f"images/{filename}.png"))
