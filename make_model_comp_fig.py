# 4つのモデルの精度の推移をplotする
# 10個のノードの精度の平均値の推移
# modelは{vgg19_bn, resnet_152, mobilenet_v2, vit_b16}

import csv
import os
import pickle

import matplotlib.pyplot as plt

project_path = "../data-raid/static/WAFL_pos_estimation"
vgg_dir_name = "vgg_wafl_raw_iid"
resnet_dir_name = "resnet_wafl_raw_iid"
mobilenet_dir_name = "mobile_wafl_raw_iid"
vit_dir_name = "vit_wafl_raw_iid"
image_file_name = "comp_WAFL_ML_rwp"
image_path = os.path.join(project_path, "model_comp")  # 出力先のFolder

# 各エポックの精度を記録した配列をloadする。
# vgg19_bn
with open(
    os.path.join(project_path, vgg_dir_name, "params", "historys_data.pkl"), "rb"
) as f_vgg:
    vgg_his = pickle.load(f_vgg)
# resnet_152
with open(
    os.path.join(project_path, resnet_dir_name, "params", "historys_data.pkl"),
    "rb",
) as f_resnet:
    resnet_his = pickle.load(f_resnet)
# mobilenet_v2
with open(
    os.path.join(project_path, mobilenet_dir_name, "params", "historys_data.pkl"),
    "rb",
) as f_mobilenet:
    mobilenet_his = pickle.load(f_mobilenet)
# vit_b16
with open(
    os.path.join(project_path, vit_dir_name, "params", "historys_data.pkl"), "rb"
) as f_vit:
    vit_his = pickle.load(f_vit)

# 10個のノードの平均を求める
# vgg19_bn
n_node = len(vgg_his)
num_epoch = len(vgg_his[0])
vgg_avg = []
for epoch in range(0, num_epoch):
    avg = 0
    for node in range(0, n_node):
        avg += vgg_his[node][epoch][4]
    avg = avg / n_node
    vgg_avg.append(avg)

# resnet_152
n_node = len(resnet_his)
num_epoch = len(resnet_his[0])
resnet_avg = []
for epoch in range(0, num_epoch):
    avg = 0
    for node in range(0, n_node):
        avg += resnet_his[node][epoch][4]
    avg = avg / n_node
    resnet_avg.append(avg)

# mobilenet_v2
n_node = len(mobilenet_his)
num_epoch = len(mobilenet_his[0])
mobilenet_avg = []
for epoch in range(0, num_epoch):
    avg = 0
    for node in range(0, n_node):
        avg += mobilenet_his[node][epoch][4]
    avg = avg / n_node
    mobilenet_avg.append(avg)

# vit_b16
n_node = len(vit_his)
num_epoch = len(vit_his[0])
vit_avg = []
for epoch in range(0, num_epoch):
    avg = 0
    for node in range(0, n_node):
        avg += vit_his[node][epoch][4]
    avg = avg / n_node
    vit_avg.append(avg)

# plotする
epoch_array = [i for i in range(0, num_epoch)]
plt.plot(epoch_array, vit_avg, label="Vision Transformer")
plt.plot(epoch_array, resnet_avg, label="ResNet")
plt.plot(epoch_array, vgg_avg, label="VGG")
plt.plot(epoch_array, mobilenet_avg, label="MobileNetV2")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.xlim([-100, 3100])
plt.ylim([0.3, 0.9])
plt.legend()
plt.savefig(os.path.join(image_path, f"{image_file_name}_acc.png"), bbox_inches="tight")
