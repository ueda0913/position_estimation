# 4つのモデルの精度の推移をplotする
# 10個のノードの精度の平均値の推移
# modelは{vgg19_bn, resnet_152, mobilenet_v2, vit_b16}

import csv
import os
import pickle

import matplotlib.pyplot as plt

# セーブするファイルのpathとloadする精度データのpath
csv_path = "Path to csv file folder"  # 出力先のFolderを指定する
image_path = "Path to image folder"  # 出力先のFolderを指定する
log_dir = (
    "/mnt/data-raid/muramatsu/data/training_log"  # GPU2. Training logのあるディレクトリのpath
)
# log_dir = 'home/muramatsu/file_for_docker/WAFL/training_log' # GPU1. logのあるディレクトリのpath

# 各エポックの精度を記録した配列をloadする。
# vgg19_bn
with open(
    os.path.join(log_dir, "WAFL_vgg19_bn_SGD_IID", "params", "histories_data.pkl"), "rb"
) as f_vgg:
    vgg_his = pickle.load(f_vgg)
# resnet_152
with open(
    os.path.join(log_dir, "WAFL_resnet_152_SGD_IID", "params", "histories_data.pkl"),
    "rb",
) as f_resnet:
    resnet_his = pickle.load(f_resnet)
# mobilenet_v2
with open(
    os.path.join(log_dir, "WAFL_mobilenet_v2_SGD_IID", "params", "histories_data.pkl"),
    "rb",
) as f_mobilenet:
    mobilenet_his = pickle.load(f_mobilenet)
# vit_b16
with open(
    os.path.join(log_dir, "WAFL_vit_b16_SGD_IID", "params", "histories_data.pkl"), "rb"
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

# csvファイルに出力する(他のアプリでグラフを書く時のため)
with open(os.path.join(csv_path, "WAFL_IID_average_accuracy.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "vgg", "resnet", "mobilenet", "vit"])
    for i in range(0, num_epoch):
        writer.writerow([i, vgg_avg[i], resnet_avg[i], mobilenet_avg[i], vit_avg[i]])

# plotする
epoch_array = [i for i in range(0, num_epoch)]
plt.plot(epoch_array, vit_avg, label="WAFL_ViT")
plt.plot(epoch_array, resnet_avg, label="WAFL_ResNet")
plt.plot(epoch_array, vgg_avg, label="WAFL_VGG")
plt.plot(epoch_array, mobilenet_avg, label="WAFL_MobileNet")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig(
    os.path.join(image_path, "WAFL_IID_average_accuracy.png"), bbox_inches="tight"
)
plt.savefig(
    os.path.join(image_path, "WAFL_IID_average_accuracy.eps"), bbox_inches="tight"
)
