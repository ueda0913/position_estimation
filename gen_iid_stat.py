# IIDデータセットの平均と分散を求める
import os

import torch
from definitions.train_functions import *
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder

# Data path
data_dir = "../data-raid/data/UTokyoE_building_dataset"
classes = ("安田講堂", "工2", "工3", "工13", "工4", "工8", "工1", "工6", "列品館", "法文1")
train_dir = os.path.join(data_dir, "train")
# 保存先
noniid_filter_dir = os.path.join(data_dir, "noniid_filter")
mean_file = os.path.join(noniid_filter_dir, "IID_train_mean.pt")
std_file = os.path.join(noniid_filter_dir, "IID_train_std.pt")

# 変数
batch_size = 16
n_node = 10

# seedの設定
torch_seed()  # seedの固定 # 場所の変更------------
# g = torch.Generator()
# g.manual_seed(0)

# datasetの用意
tmp_transform = transforms.Compose(  # Non-IIdフィルタと合わせた
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)
train_data = ImageFolder(train_dir, transform=tmp_transform)

# IIDでデータセットを分割
indices = [[] for i in range(n_node)]
for i in range(len(train_data)):
    indices[i % n_node].append(i)

# Assign training data to each node
# for i in range(n_node):# データ分布の出力
#     print(f"node_{i}:{indices[i]}\n")
subset = [Subset(train_data, indices[i]) for i in range(n_node)]
nums = [[0 for i in range(n_node)] for j in range(n_node)]
# for i in range(n_node): # データ分布の出力を行う
#     for j in range(len(subset[i])):
#         image, label = subset[i][j]
#         nums[i][int(label)] += 1
#     print(f'Distributions of data')
#     print(f"train_data of node_{i}: {nums[i]}\n")

train_loader = [
    DataLoader(subset[i], batch_size=batch_size, num_workers=50)
    for i in range(0, n_node)
]
# for i in range(0, n_node):
#     train_loader.append(DataLoader(subset[i], batch_size=batch_size, shuffle=True))

print(len(subset[0]))
print(len(train_loader[0]))
means = [torch.zeros(3) for i in range(n_node)]
stds = [torch.zeros(3) for i in range(n_node)]

# dataloaderから読み出して平均と分散を求める
for i in range(0, n_node):  # nodeに関してforループ
    for data in train_loader[i]:
        images, labels = data
        batch_size = len(labels)
        labels = labels.tolist()
        for j in range(len(labels)):  # 一つのバッチ内のデータを取り出す
            means[i] += images[j].mean(dim=(1, 2))  # node_iの平均配列に加算
            stds[i] += images[j].std(dim=(1, 2))

for i in range(n_node):
    means[i] /= len(indices[i])  # node_iの持つデータ数で割る
    stds[i] /= len(indices[i])

print(means)
print(stds)
torch.save(means, mean_file)
torch.save(stds, std_file)
