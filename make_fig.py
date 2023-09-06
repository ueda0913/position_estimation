import os

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from definitions.mydatasets import MyGPUdataset
from definitions.net import select_net
from definitions.visualize import show_image_labels
from torch.utils.data import DataLoader
from torchvision import models
from train_pos_estimation import (
    data_dir,
    model_name,
    n_middle,
    n_node,
    project_path,
    useGPUinTrans,
)

###change area
epoch = 3000
static_date_index = "2023-09-03-09"  # trained epoch to load
batch_size = 16
all_images = False
node = 0  # node num


classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

meant_file_path = os.path.join(data_dir, "test_mean.pt")
stdt_file_path = os.path.join(data_dir, "test_std.pt")
mean_t = torch.load(meant_file_path)
std_t = torch.load(stdt_file_path)
print("loading of mean and std in test data finished")

if useGPUinTrans:
    pre_transform = transforms.Resize(256)
    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())
            ),
        ]
    )
    test_data = MyGPUdataset(
        test_dir,
        device,
        len(classes),
        transform=test_transform,
        pre_transform=pre_transform,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
else:
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())
            ),
        ]
    )
    test_data = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=not (all_images),
        num_workers=60,
        pin_memory=True,
    )

si_dir_path = os.path.join(project_path, static_date_index, "images/sample_images")
if not (os.path.exists(si_dir_path)) or os.path.isfile(si_dir_path):
    os.makedirs(si_dir_path)

if __name__ == "__main__":
    nets = [
        select_net(model_name, len(classes), n_middle).to(device) for i in range(n_node)
    ]
    nets.load_state_dict(
        torch.load(
            os.path.join(
                project_path, static_date_index, f"params/epoch-{epoch:04d}.pth"
            )
        )
    )
    show_image_labels(
        test_loader,
        classes,
        nets[node],
        device,
        epoch,
        static_date_index,
        project_path,
        all_images,
    )
