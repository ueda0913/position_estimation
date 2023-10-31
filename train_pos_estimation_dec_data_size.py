import copy
import json
import os
import pickle
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from definitions.make_log import *
from definitions.model_exchange import *
from definitions.mydataset import *
from definitions.net import *
from definitions.train_functions import *
from definitions.visualize import *
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchinfo import summary
from torchvision.datasets import ImageFolder

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

# path
data_dir = "../data-raid/data/position_estimation_dataset"
# data_dir = "../data-raid/data/UTokyoE_building_dataset"
project_path = "../data-raid/static/WAFL_pos_estimation"
noniid_filter_dir = os.path.join(data_dir, "noniid_filter")
contact_pattern_dir = "../data-raid/static/contact_pattern"
classes = ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")
meant_file_path = os.path.join(data_dir, "test_mean.pt")
stdt_file_path = os.path.join(data_dir, "test_std.pt")


### change area
## about training conditions
cur_time_index = datetime.now().strftime("%Y-%m-%d-%H")
# cur_time_index = "2023-10-25-10"
device = torch.device(
    "cuda:1" if torch.cuda.is_available() else "cpu"
)  # use 0 in GPU1 use 1 in GPU2
max_epoch = 3000
pre_train_epoch = 150
batch_size = 16
n_node = 12
n_middle = 256
model_name = "vit_b16"  # vgg19_bn or mobilenet_v2 or resnet_152 or vit_b16
optimizer_name = "SGD"  # SGD or Adam
lr = 0.05
momentum = 0.9
pretrain_lr = 0.05
pretrain_momentum = 0.9

# cos similarity
use_cos_similarity = False
st_fl_coefficiency = 0.1  # 使わない場合の値
sat_epoch = 2500  # cos類似度を使わなくなるepoch

# schedulers
use_scheduler = True  # if do not use scheduler, False here
scheduler_step = 750
scheduler_rate = 0.3
use_pretrain_scheduler = True
pretrain_scheduler_step = 50
pretrain_scheduler_rate = 0.3

## about the data each node have
is_use_noniid_filter = False
filter_rate = 70
filter_seed = 1

## about contact patterns
contact_file = "rwp_n12_a0500_r100_p40_s01.json"
# contact_file = "static_line_n12.json"
# contact_file=f'cse_n10_c10_b02_tt05_tp2_s01.json'
# contact_file = 'meet_at_once_t10000.json'

## select train mode
use_previous_memory = False  # use the past memory
is_pre_train_only = False  # use to do only pre-training
is_train_only = False  # use to load pre-trained data and start training from scratch
is_restart = False  # use to load traied_data and add training
load_time_index = (
    None  # use when "is_train_only" or "is_restart" flag is valid. check situation
)
load_epoch = None  # use when "is_restart" flag is valid. how many epochs the model trained in the former training

cur_dir = os.path.join(project_path, cur_time_index)
contact_file_path = os.path.join(contact_pattern_dir, contact_file)

torch_seed()
g = torch.Generator()
g.manual_seed(123)

print("using device", device)
schedulers = None
pretrain_schedulers = None

# make test_transform
if not (os.path.exists(meant_file_path)) or not (os.path.exists(stdt_file_path)):
    mean_t, std_t = search_mean_and_std(test_dir)
    torch.save(mean_t, meant_file_path)
    torch.save(std_t, stdt_file_path)
    print("calculation of mean and std in test data finished")
else:
    mean_t = torch.load(meant_file_path)
    std_t = torch.load(stdt_file_path)
    print("loading of mean and std in test data finished")

test_transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=tuple(mean_t.tolist()), std=tuple(std_t.tolist())),
    ]
)

# makde train_data
train_data = MyGPUdataset(
    train_dir, device, len(classes), pre_transform=transforms.Resize(256)
)
# train_data = ImageFolder(train_dir)
test_data = MyGPUdataset(
    test_dir,
    device,
    len(classes),
    transform=test_transform,
    pre_transform=transforms.Resize(256),
)


# loading filter and statistics
if is_use_noniid_filter:
    means = torch.load(
        os.path.join(
            noniid_filter_dir, f"mean_r{filter_rate:02d}_s{filter_seed:02d}.pt"
        )
    )
    stds = torch.load(
        os.path.join(noniid_filter_dir, f"std_r{filter_rate:02d}_s{filter_seed:02d}.pt")
    )
    print("Loading of mean and std in non-iid train data finished")

    # loading filter file or not
    filter_file = f"filter_r{filter_rate:02d}_s{filter_seed:02d}.pt"
    indices = torch.load(os.path.join(noniid_filter_dir, filter_file))

else:
    filter_file = None
    indices = [[] for i in range(n_node)]
    for i in range(len(train_data)):
        indices[i % n_node].append(i)

    means = torch.load(os.path.join(noniid_filter_dir, "IID_train_mean.pt"))
    stds = torch.load(os.path.join(noniid_filter_dir, "IID_train_std.pt"))
    print("Loading of mean and std in iid train data finished")

# set train data into subset
for i in range(n_node):
    print(f"node_{i}:{indices[i]}\n")
subset = [Subset(train_data, indices[i]) for i in range(n_node)]

# print data distribution
nums = [[0 for i in range(n_node)] for j in range(n_node)]
for i in range(n_node):
    for j in range(len(subset[i])):
        image, label = subset[i][j]
        nums[i][int(label)] += 1
    print(f"train_data of node_{i}: {nums[i]}\n")


# make train_data_loader
trainloader = []
for i in range(len(subset)):
    mean = means[i]
    mean = mean.tolist()
    std = stds[i]
    std = std.tolist()
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=224, scale=(0.4, 1.0)),
            # transforms.RandomCrop(224),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
            # transforms.Normalize(0.5, 0.5)
            transforms.RandomErasing(
                p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False
            ),
        ]
    )
    train_dataset_new = FromSubsetDataset(
        subset[i],
        transform=train_transform,
    )
    trainloader.append(
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

# make test_dataloader
testloader = DataLoader(
    test_data,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    worker_init_fn=seed_worker,
    generator=g,
)

# define net, optimizer, criterion
criterion = nn.CrossEntropyLoss()

nets = [
    select_net(model_name, len(classes), n_middle).to(device) for i in range(n_node)
]
optimizers = [
    select_optimizer(model_name, nets[i], optimizer_name, lr, momentum)
    for i in range(n_node)
]
pretrain_optimizers = [
    select_optimizer(
        model_name, nets[i], optimizer_name, pretrain_lr, pretrain_momentum
    )
    for i in range(n_node)
]

if use_scheduler:
    schedulers = [
        optim.lr_scheduler.StepLR(
            optimizers[i], step_size=scheduler_step, gamma=scheduler_rate
        )
        for i in range(n_node)
    ]
if use_pretrain_scheduler:
    pretrain_schedulers = [
        optim.lr_scheduler.StepLR(
            pretrain_optimizers[i],
            step_size=pretrain_scheduler_step,
            gamma=pretrain_scheduler_rate,
        )
        for i in range(n_node)
    ]

contact_list = []
historys = [np.zeros((0, 5)) for i in range(n_node)]
pre_train_historys = [np.zeros((0, 5)) for i in range(n_node)]
if __name__ == "__main__":
    if is_train_only:
        with open(
            os.path.join(project_path, load_time_index, "params", "historys_data.pkl"),
            "rb",
        ) as f:
            historys = pickle.load(f)
        for n in range(n_node):
            nets[n].load_state_dict(
                torch.load(
                    os.path.join(
                        project_path, f"{load_time_index}/params/Pre-train-node{n}.pth"
                    )
                )
            )
        cur_time_index = load_time_index
        cur_dir = os.path.join(project_path, cur_time_index)
        load_epoch = 0
        print("Just training from pre-train result")
        with open(os.path.join(cur_dir, "log.txt"), "a") as f:
            f.write(
                "train under the following conditions. confirm them to be same as that of pre-train\n"
            )

    elif is_restart:
        with open(
            os.path.join(project_path, load_time_index, "params", "historys_data.pkl"),
            "rb",
        ) as f:
            historys = pickle.load(f)
        if len(historys[0]) != load_epoch + 1:
            print("error: do not load suitable file")
            exit(1)
        for n in range(n_node):
            nets[n].load_state_dict(
                torch.load(
                    os.path.join(
                        project_path,
                        f"{load_time_index}/params/node{n}_epoch-{load_epoch:04d}.pth",
                    )
                )
            )
        cur_time_index = load_time_index
        cur_dir = os.path.join(project_path, cur_time_index)
        print(f"restart training from epoch{load_epoch+1}")
        with open(os.path.join(cur_dir, "log.txt"), "a") as f:
            f.write(
                "restart training under the following conditions. confirm them to be same as that of pre-train\n"
            )

    else:
        os.makedirs(cur_dir)
        show_dataset_contents(data_dir, classes, cur_dir)

    initial_log(
        cur_dir,
        subset,
        batch_size,
        contact_file,
        is_use_noniid_filter,
        filter_file,
        testloader,
        trainloader,
        optimizers,
        use_scheduler,
        schedulers,
        pretrain_optimizers,
        pretrain_schedulers,
        use_pretrain_scheduler,
        use_previous_memory,
        use_cos_similarity,
        st_fl_coefficiency,
        is_pre_train_only,
        nets,
    )

    if (not is_train_only) and (not is_restart):
        os.makedirs(os.path.join(cur_dir, "params"))
        load_epoch = 0
        # pre-self training
        pre_train(
            nets,
            trainloader,
            testloader,
            pretrain_optimizers,
            criterion,
            pre_train_epoch,
            device,
            cur_dir,
            historys,
            pre_train_historys,
            pretrain_schedulers,
        )

        history_save_path = os.path.join(cur_dir, "params", "historys_data.pkl")
        with open(history_save_path, "wb") as f:
            pickle.dump(historys, f)
        print("saving historys...")
        pre_train_history_save_path = os.path.join(
            cur_dir, "params", "pre_train_historys_data.pkl"
        )
        with open(pre_train_history_save_path, "wb") as f:
            pickle.dump(pre_train_historys, f)
            print("saving pre_train historys...")

        if is_pre_train_only:
            mean, std = calc_res_mean_and_std(pre_train_historys)
            with open(os.path.join(cur_dir, "log.txt"), "a") as f:
                f.write(f"the average of the last 10 epoch: {mean}\n")
                f.write(f"the std of the last 10 epoch: {std}\n")

            exit(0)

    # load contact pattern
    print(f"Loading ... {contact_file_path}")
    with open(contact_file_path) as f:
        contact_list = json.load(f)

    # below 3 rows are used to use previous memory
    if use_previous_memory:
        former_contact = {str(i): [] for i in range(n_node)}
        former_nets = [{} for _ in range(n_node)]
        counters = [0 for _ in range(n_node)]
        former_exchange_num = [
            0 for _ in range(n_node)
        ]  # how many times exchange with former one

    for epoch in range(
        load_epoch, max_epoch + load_epoch
    ):  # loop over the dataset multiple times
        contact = contact_list[epoch]
        # below row are used to use previous memory(now only for vit)
        if use_previous_memory:
            model_exchange_with_former(
                former_contact,
                contact,
                former_nets,
                nets,
                counters,
                former_exchange_num,
                model_name,
            )
        model_exchange(
            nets,
            model_name,
            contact,
            use_cos_similarity,
            st_fl_coefficiency,
            epoch,
            sat_epoch,
        )

        for n in range(n_node):
            nbr = contact[str(n)]
            if len(nbr) == 0:
                item = np.array(
                    [
                        epoch + 1,
                        historys[n][-1][1],
                        historys[n][-1][2],
                        historys[n][-1][3],
                        historys[n][-1][4],
                    ]
                )
                historys[n] = np.vstack((historys[n], item))
                print(
                    f"Epoch [{epoch+1}], Node [{n}], loss: {historys[n][-1][1]:.5f} acc: {historys[n][-1][2]:.5f} val_loss: {historys[n][-1][3]:.5f} val_acc: {historys[n][-1][4]:.5f}"
                )
            else:
                historys[n] = fit(
                    nets[n],
                    optimizers[n],
                    criterion,
                    trainloader[n],
                    testloader,
                    device,
                    historys[n],
                    epoch,
                    n,
                )

            # make figs
            if (
                epoch > ((max_epoch + load_epoch) * 0.8) and epoch % 50 == 49
            ) or epoch == max_epoch + load_epoch - 1:
                train_for_cmls(
                    cur_dir,
                    epoch,
                    n,
                    classes,
                    nets[n],
                    criterion,
                    testloader,
                    device,
                )

            # write log
            if epoch % 100 == 99 or epoch + 10 > max_epoch + load_epoch - 1:
                with open(os.path.join(cur_dir, "log.txt"), "a") as f:
                    f.write(
                        f"Epoch [{epoch+1}], Node [{n}], loss: {historys[n][-1][1]:.5f} acc: {historys[n][-1][2]:.5f} val_loss: {historys[n][-1][3]:.5f} val_acc: {historys[n][-1][4]:.5f}\n"
                    )

            # save histories
            if epoch % 500 == 499:
                history_save_path = os.path.join(cur_dir, "params", "historys_data.pkl")
                with open(history_save_path, "wb") as f:
                    pickle.dump(historys, f)
                    print("saving historys...")

                torch.save(
                    nets[n].state_dict(),
                    os.path.join(cur_dir, f"params/node{n}_while_training.pth"),
                )
                torch.save(
                    optimizers[n].state_dict(),
                    os.path.join(
                        cur_dir, f"params/node{n}_optimizer_while_training.pth"
                    ),
                )
                print(f"Model saving ... at {epoch+1}")
                nets[n] = nets[n].to(device)

            # save models
            if epoch == max_epoch + load_epoch - 1:
                if os.path.exists(
                    os.path.join(cur_dir, f"params/node{n}_while_training.pth")
                ):
                    os.remove(
                        os.path.join(cur_dir, f"params/node{n}_while_training.pth")
                    )
                if os.path.exists(
                    os.path.join(
                        cur_dir, f"params/node{n}_optimizer_while_training.pth"
                    )
                ):
                    os.remove(
                        os.path.join(
                            cur_dir, f"params/node{n}_optimizer_while_training.pth"
                        )
                    )

                print(f"Model saving ... at {epoch+1}")
                torch.save(
                    nets[n].state_dict(),
                    os.path.join(cur_dir, f"params/node{n}_epoch-{epoch+1:04d}.pth"),
                )
                nets[n] = nets[n].to(device)
                torch.save(
                    optimizers[n].state_dict(),
                    os.path.join(
                        cur_dir, f"params/node{n}_optimizer_epoch-{epoch+1:04d}.pth"
                    ),
                )
                if use_previous_memory:
                    print(
                        f"former exchange: {former_exchange_num}"
                    )  # to confirm how many times exchange with former one

            # update scheduler
            if schedulers != None:
                schedulers[n].step()

    history_save_path = os.path.join(cur_dir, "params", "historys_data.pkl")
    with open(history_save_path, "wb") as f:
        pickle.dump(historys, f)
        print("saving historys...")
    mean, std, max_acc, min_acc = calc_res_mean_and_std(historys)
    with open(os.path.join(cur_dir, "log.txt"), "a") as f:
        f.write(f"the average of the last 10 epoch: {mean}\n")
        f.write(f"the std of the last 10 epoch: {std}\n")
        f.write(f"the maxmize of the last 10 epoch: {max_acc}\n")
        f.write(f"the minimum of the last 10 epoch: {min_acc}\n")
        f.write(f"Usage of previous memory: {former_exchange_num}\n")
    evaluate_history(historys, cur_dir)
    print("Finished Training")
