import os

import torch
import torchvision.transforms as transforms
from definitions.mydataset import MyGPUdataset
from definitions.net import select_net
from definitions.visualize import train_for_cmls
from torch.utils.data import DataLoader

## change area
epoch = 3000
node = 6
batch_size = 16
model_name = "vit_b16"  # vgg19_bn or mobilenet_v2 or resnet_152 or vit_b16
load_dir_name = "vit_wafl_raw_iid_line"
n_middle = 256
criterion = torch.nn.CrossEntropyLoss()


classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
project_path = "../data-raid/static/WAFL_pos_estimation"
data_dir = "../data-raid/data/position_estimation_dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")
cur_dir = os.path.join(project_path, load_dir_name)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

meant_file_path = os.path.join(data_dir, "test_mean.pt")
stdt_file_path = os.path.join(data_dir, "test_std.pt")
mean_t = torch.load(meant_file_path)
std_t = torch.load(stdt_file_path)
print("loading of mean and std in test data finished")

pre_transform = transforms.Resize(256)
test_transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())),
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

model_path = os.path.join(cur_dir, f"params/node{node}_epoch-{epoch:04d}.pth")
net = select_net(model_name, len(classes), n_middle).to(device)
net.load_state_dict(torch.load(model_path))
train_for_cmls(cur_dir, epoch - 1, node, classes, net, criterion, test_loader, device)
