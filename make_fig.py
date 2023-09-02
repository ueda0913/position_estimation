import json
import os
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from definitions import (
    FromSubsetDataset,
    MyGPUdataset,
    MyGPUdatasetFolder,
    search_mean_and_std,
    torch_seed,
)
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchinfo import summary


class MyCPUdataset_forimg(MyGPUdataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, "cpu", transform, pre_transform)


def show_image_labels(loader):
    nums = [0 for i in range(10)]
    index = 0
    if not (os.path.isdir("/home/ueda/img_test")):
        os.makedirs("/home/ueda/img_test")
    print("make img_dir")
    for j in range(len(loader)):
        for images, labels in loader[j]:
            n_size = len(images)
            for i in range(n_size):
                label = int(labels[i])
                if nums[label] < 5:
                    nums[label] += 1
                    img = torchvision.transforms.functional.to_pil_image(
                        images[i].to("cpu")
                    )
                    img.save(f"/home/ueda/img_test/node_{label}_{nums[label]}.png")
                    index += 1
                if index >= 5 * 10:
                    return
    return 0


# path
data_dir = "../data-raid/data/UTokyoE_building_dataset"
project_path = "../data-raid/static/WAFL_research"
noniid_filter_dir = os.path.join(data_dir, "noniid_filter")
contact_pattern_dir = "../data-raid/static/contact_pattern"
classes = ("安田講堂", "工2", "工3", "工13", "工4", "工8", "工1", "工6", "列品館", "法文1")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)  # use 0 in GPU1 use 1 in GPU2


### change area
## about training conditions
cur_time_index = datetime.now().strftime("%Y-%m-%d-%H")
# cur_time_index = '2023-06-10-21'
max_epoch = 100
pre_train_epoch = 10
batch_size = 16
n_node = 10
fl_coefficiency = 0.1
model_name = "mobile"  # vgg or mobile or res or vit
isfine = False  # True if use fine-tuning
useGPUinTrans = True  # whether use GPU in transform or not
lr = 0.01
pre_train_lr = 0.01

# schedulers
use_scheduler = False  # if do not use scheduler, False here
scheduler_step = 1000
scheduler_rate = 0.5
use_pre_train_scheduler = False
pre_train_scheduler_step = 30
pre_train_scheduler_rate = 0.3

## about the data each node have
is_use_noniid_filter = True
filter_rate = 50
filter_seed = 1

## about contact patterns
contact_file = "rwp_n10_a0500_r100_p10_s01.json"
# contact_file=f'cse_n10_c10_b02_tt05_tp2_s01.json'
# contact_file = 'meet_at_once_t10000.json'

## select train mode
cur_dir = os.path.join(project_path, cur_time_index)
is_pre_train_only = False  # use to do only pre-training
is_train_only = False  # use to load pre-trained data and start training from scratch
is_restart = False  # use to load traied_data and add training
load_time_index = (
    None  # use when "is_train_only" or "is_restart" flag is valid. check situation
)
load_epoch = (
    None  # use when "is_restart" flag is valid. how many epochs shoud it be added
)


torch_seed()
print("using device", device)
schedulers = None
pre_train_schedulers = None

# make test_transform
meant_file_path = os.path.join(data_dir, "test_mean.pt")
stdt_file_path = os.path.join(data_dir, "test_std.pt")
contact_file_path = os.path.join(contact_pattern_dir, contact_file)

if not (os.path.exists(meant_file_path)) or not (os.path.exists(stdt_file_path)):
    mean_t, std_t = search_mean_and_std(test_dir)
    torch.save(mean_t, meant_file_path)
    torch.save(std_t, stdt_file_path)
else:
    mean_t = torch.load(meant_file_path)
    std_t = torch.load(stdt_file_path)
print("calculation of mean and std in test data finished")

test_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())),
    ]
)

if useGPUinTrans:
    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())
            ),
        ]
    )

train_data = None
test_data = None
if useGPUinTrans:
    train_data = MyCPUdataset_forimg(train_dir, pre_transform=transforms.Resize(256))
    test_data = MyCPUdataset_forimg(
        train_dir,
        transform=test_transform,
        pre_transform=transforms.Resize(256),
    )
else:
    train_data = datasets.ImageFolder(train_dir)
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

# loading filter file or not
filter_file = f"filter_r{filter_rate:02d}_s{filter_seed:02d}.pt"
indices = torch.load(os.path.join(noniid_filter_dir, filter_file))
if not is_use_noniid_filter:
    indices = [[] for i in range(n_node)]
    for i in range(len(train_data)):
        indices[i % n_node].append(i)

# set train data into subset
subset = [Subset(train_data, indices[i]) for i in range(10)]
means = torch.load(
    os.path.join(noniid_filter_dir, f"mean_r{filter_rate:02d}_s{filter_seed:02d}.pt")
)
stds = torch.load(
    os.path.join(noniid_filter_dir, f"std_r{filter_rate:02d}_s{filter_seed:02d}.pt")
)
print("Loading of mean and std in train data finished")

# make train_data_loader
trainloader = []
for i in range(len(subset)):
    mean = means[i]
    mean = mean.tolist()
    std = stds[i]
    std = std.tolist()
    train_transform = None
    if useGPUinTrans:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
                transforms.ConvertImageDtype(torch.float32),
                # transf orms.Normalize(mean=tuple(mean), std=tuple(std)),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=tuple(mean), std=tuple(std)),
                transforms.RandomErasing(
                    p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
                ),
            ]
        )
    train_dataset_new = FromSubsetDataset(subset[i], transform=train_transform)
    if useGPUinTrans:
        trainloader.append(
            DataLoader(
                train_dataset_new,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
        )
    else:
        trainloader.append(
            DataLoader(
                train_dataset_new,
                batch_size=batch_size,
                shuffle=True,
                num_workers=50,
                pin_memory=True,
            )
        )

# make test_dataloader
testloader = None
if useGPUinTrans:
    testloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False
    )
else:
    testloader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=50, pin_memory=True
    )

show_image_labels(trainloader)
