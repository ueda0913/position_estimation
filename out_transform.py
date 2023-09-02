import torchvision.transforms as transforms
import os
import time
import torchvision.datasets as datasets
import torch
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader
from definitions import FromSubsetDataset, MyGPUdatasetFolder

data_dir = '../data-raid/data/UTokyoE_building_dataset'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')

noniid_filter_dir = os.path.join(data_dir, 'noniid_filter')
n_node = 10
filter_rate = 10
filter_seed = 1
batch_size = 20
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(device)


# start_time = time.perf_counter()
means = torch.load(os.path.join(noniid_filter_dir, f'mean_r{filter_rate:02d}_s{filter_seed:02d}.pt'))
stds = torch.load(os.path.join(noniid_filter_dir, f'std_r{filter_rate:02d}_s{filter_seed:02d}.pt'))
meant_file_path = os.path.join(data_dir, 'test_mean.pt')
stdt_file_path = os.path.join(data_dir, 'test_std.pt')
mean_t = torch.load(meant_file_path)
std_t = torch.load(stdt_file_path)
print("calculation of mean and std in test data finished")

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # transforms.ToTensor(),
    transforms.Normalize(mean = tuple(mean_t.tolist()), std = tuple(std_t.tolist())),
])
print("Loading of mean and std in train data finished")

start_time = time.perf_counter()
# train_data = datasets.ImageFolder(train_dir, transform=transforms.Compose([transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)), transforms.ToTensor()]))
# train_data = MyGPUdatasetFolder(train_dir, device)
# # train_data = datasets.ImageFolder(train_dir)
# indices = [[] for i in range(n_node)]
# for i in range(len(train_data)):
#     indices[i%n_node].append(i)
# subset=[Subset(train_data, indices[i]) for i in range(10)]

# trainloader = []
# for i in range(len(subset)):
#     mean = means[i]
#     mean = mean.tolist()
#     std = stds[i]
#     std = std.tolist()
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
#         # transforms.ToTensor(),
#         transforms.Normalize(mean = tuple(mean), std = tuple(std)),
#         transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)
#     ])
#     train_dataset_new = FromSubsetDataset(subset[i], transforms=train_transform)
#     # print(subset[i][0][0], subset[i][0][1])
#     # train_dataset_new = FromSubsetDatasetGPU(subset[i], device, mytransforms=train_transform)
#     # print(i)
#     # print(f'node{i} data for train: {len(train_dataset_new)}')
#     trainloader.append(DataLoader(train_dataset_new, batch_size=batch_size,
#                     shuffle=True, num_workers=0, pin_memory=False))
# i = 0
# for image, _ in trainloader[0]:
#     i += 1
# print(i)

test_data = MyGPUdatasetFolder(val_dir, device, transform=test_transform)
# test_data = datasets.ImageFolder(val_dir, transform=test_transform)
testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
for image, label in testloader:
    continue
end_time = time.perf_counter()
print(end_time-start_time)