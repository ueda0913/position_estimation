import os
import random

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from definitions.mydataset import *
from definitions.train_functions import *
from torch.utils.data.dataset import Subset

batch_size = 16
n_node = 12
ratio = 70  # the rate that n-th node has n-labeled picture
data_dir = "../data-raid/data/position_estimation_dataset"
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)  # use 0 in GPU1 use 1 in GPU2

# seedの設定
torch_seed()  # seedの固定 # 場所の変更------------
g = torch.Generator()
g.manual_seed(123)
randomseed = 2
random.seed(randomseed)


filename = os.path.join(
    data_dir, f"noniid_filter/filter_r{ratio:02d}_s{randomseed:02d}.pt"
)
meanfile = os.path.join(
    data_dir, f"noniid_filter/mean_r{ratio:02d}_s{randomseed:02d}.pt"
)
stdfile = os.path.join(data_dir, f"noniid_filter/std_r{ratio:02d}_s{randomseed:02d}.pt")
print(f"Generating NonIID filter ... {filename}")

train_dir = os.path.join(data_dir, "train")

tmp_transform = transforms.Compose(
    [
        transforms.ConvertImageDtype(torch.float32),
    ]
)
train_data = MyGPUdataset(
    train_dir,
    device,
    n_node,
    transform=tmp_transform,
    pre_transform=transforms.Resize(256),
)
trainloader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    num_workers=0,
    shuffle=False,
    pin_memory=False,
    worker_init_fn=seed_worker,
    generator=g,
)

indices = [[] for _ in range(n_node)]  # indices[i]はi番目のノードのデータ

means = [torch.zeros(3).to(device) for i in range(n_node)]
stds = [torch.zeros(3).to(device) for i in range(n_node)]

index = 0
for data in trainloader:
    x, y = data
    batch_size = len(y)
    y = y.tolist()

    for i in range(len(y)):
        if random.randint(0, 99) < ratio:
            indices[y[i]].append(index + i)
            means[y[i]] += x[i].mean(dim=(1, 2))
            stds[y[i]] += x[i].std(dim=(1, 2))
        else:
            n = random.randint(0, 8)
            if y[i] <= n:
                n += 1
            indices[n].append(index + i)
            means[n] += x[i].mean(dim=(1, 2))
            stds[n] += x[i].std(dim=(1, 2))

    index += batch_size

for i in range(len(indices)):
    means[i] /= len(indices[i])
    stds[i] /= len(indices[i])
    means[i] = means[i].to("cpu")
    stds[i] = stds[i].to("cpu")
    print(f"node_{i}:{indices[i]}\n")

print(f"means:\n{means}\nstds:\n{stds}")
torch.save(indices, filename)
torch.save(means, meanfile)
torch.save(stds, stdfile)
print("Done")

# for checking
# subset = Subset(trainset,indices[1])
# subloader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
#                                          num_workers=2)
# for data in subloader :
#     x, y= data
#     print(y)
