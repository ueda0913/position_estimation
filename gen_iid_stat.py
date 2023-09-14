# IIDデータセットの平均と分散を求める
import os

import torch
from definitions.mydataset import *
from definitions.train_functions import *
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder

# Data path
data_dir = "../data-raid/data/position_estimation_dataset"
classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
train_dir = os.path.join(data_dir, "train")
# 保存先
noniid_filter_dir = os.path.join(data_dir, "noniid_filter")
mean_file = os.path.join(noniid_filter_dir, "IID_train_mean.pt")
std_file = os.path.join(noniid_filter_dir, "IID_train_std.pt")
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)  # use 0 in GPU1 use 1 in GPU2

# 変数
batch_size = 16
n_node = 12

# seedの設定
torch_seed()  # seedの固定 # 場所の変更------------
g = torch.Generator()
g.manual_seed(123)

# datasetの用意
tmp_transform = transforms.Compose(  # Non-IIdフィルタと合わせた
    [
        transforms.ConvertImageDtype(torch.float32),
    ]
)
train_data = MyGPUdataset(
    train_dir, device, len(classes), pre_transform=transforms.Resize(256)
)

# IIDでデータセットを分割
indices = [[] for i in range(n_node)]
for i in range(len(train_data)):
    indices[i % n_node].append(i)

# Assign training data to each node
# for i in range(n_node):# データ分布の出力
#     print(f"node_{i}:{indices[i]}\n")
subset = [Subset(train_data, indices[i]) for i in range(n_node)]
nums = [[0 for i in range(n_node)] for j in range(n_node)]
for i in range(n_node):  # データ分布の出力を行う
    for j in range(len(subset[i])):
        image, label = subset[i][j]
        nums[i][int(label)] += 1
    print(f"Distributions of data")
    print(f"train_data of node_{i}: {nums[i]}\n")

train_loader = []
for i in range(0, n_node):
    train_dataset_new = FromSubsetDataset(
        subset[i],
        transform=tmp_transform,
    )
    train_loader.append(
        DataLoader(
            train_dataset_new,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            worker_init_fn=seed_worker,
            generator=g,
        )
    )

print(len(subset[0]))
print(len(train_loader[0]))
means = [torch.zeros(3).to(device) for i in range(n_node)]
stds = [torch.zeros(3).to(device) for i in range(n_node)]

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
    means[i] = means[i].to("cpu")
    stds[i] = stds[i].to("cpu")

print(means)
print(stds)
torch.save(means, mean_file)
torch.save(stds, std_file)
