import gc
import json
import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from definitions.make_log import *
from definitions.model_exchange import *
from definitions.mydataset import *
from definitions.net import *
from definitions.train_functions import *
from definitions.visualize import *
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

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

cur_time_index = datetime.now().strftime("%Y-%m-%d-%H")
# cur_time_index = "vit_wafl_raw_noniid"
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
lr = 0.005
momentum = 0.9
pretrain_lr = 0.005
pretrain_momentum = 0.9

# cos similarity
use_cos_similarity = False
sat_epoch = 2500  # cos類似度を使わなくなるepoch

# schedulers
use_scheduler = False  # if do not use scheduler, False here
scheduler_step = 750
scheduler_rate = 0.3
use_pretrain_scheduler = False
pretrain_scheduler_step = 50
pretrain_scheduler_rate = 0.3

## about the data each node have
is_use_noniid_filter = True
filter_rate = 70
filter_seed = 1

## about contact patterns
contact_file = "rwp_n12_a0500_r100_p40_s01.json"
# contact_file = "static_line_n12.json"
# contact_file=f'cse_n10_c10_b02_tt05_tp2_s01.json'
# contact_file = 'meet_at_once_t10000.json'

## 事前学習したモデルを必ず使用すること。そのディレクトリ名を記述
use_previous_memory = False  # use the past memory
is_pre_train_only = False  # use to do only pre-training
load_dir_name = "pre_train_p40_e150"
load_epoch = 0  # use when "is_restart" flag is valid. how many epochs the model trained in the former training

base_cur_dir = os.path.join(project_path, cur_time_index)
loading_dir = os.path.join(project_path, load_dir_name)
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
trainloader = make_trainloader(subset, means, stds, batch_size, g)

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


os.makedirs(base_cur_dir)
show_dataset_contents(data_dir, classes, base_cur_dir)
initial_log_for_tuning(
    base_cur_dir,
    subset,
    batch_size,
    contact_file,
    is_use_noniid_filter,
    filter_file,
    testloader,
    trainloader,
    optimizer_name,
    use_scheduler,
    schedulers,
    pretrain_schedulers,
    use_pretrain_scheduler,
    use_previous_memory,
    use_cos_similarity,
    model_name,
    load_dir_name,
)


def objective(trial):
    # optunaによる最適化で動かすハイパーパラメータ
    st_fl_coefficiency = trial.suggest_float("st_fl_coefficiency", 0.1, 0.5, log=True)

    # define net, optimizer, criterion
    criterion = nn.CrossEntropyLoss()
    load_epoch = 0

    with open(
        os.path.join(loading_dir, "params", "historys_data.pkl"),
        "rb",
    ) as f:
        historys = pickle.load(f)

    nets = [
        select_net(model_name, len(classes), n_middle).to(device) for i in range(n_node)
    ]
    for n in range(n_node):
        nets[n].load_state_dict(
            torch.load(os.path.join(loading_dir, "params", f"Pre-train-node{n}.pth"))
        )

    optimizers = [
        select_optimizer(model_name, nets[i], optimizer_name, lr, momentum)
        for i in range(n_node)
    ]

    if use_scheduler:
        schedulers = [
            optim.lr_scheduler.StepLR(
                optimizers[i], step_size=scheduler_step, gamma=scheduler_rate
            )
            for i in range(n_node)
        ]
    contact_list = []

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
            if epoch % 100 == 99 or max_epoch - epoch < 10:
                print(
                    f"Epoch [{epoch+1}], Node [{n}], loss: {historys[n][-1][1]:.5f} acc: {historys[n][-1][2]:.5f} val_loss: {historys[n][-1][3]:.5f} val_acc: {historys[n][-1][4]:.5f}"
                )

            # update scheduler
            if schedulers != None:
                schedulers[n].step()
    # save logs
    mean, std, max_acc, min_acc = calc_res_mean_and_std(historys)
    with open(os.path.join(base_cur_dir, "log.txt"), "a") as f:
        f.write(f"--------------\nfl_coefficiency={st_fl_coefficiency}\n")
        f.write(f"the average of the last 10 epoch: {mean}\n")
        f.write(f"the std of the last 10 epoch: {std}\n")
        f.write(f"the maxmize of the last 10 epoch: {max_acc}\n")
        f.write(f"the minimum of the last 10 epoch: {min_acc}\n")
        if use_previous_memory:
            f.write(f"Usage of previous memory: {former_exchange_num}\n")
    return mean


study_name = f"study-database"
storage = f"sqlite:///{base_cur_dir}/log.db"
for i in range(20):
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)
    best_trial = study.best_trial.number
    current_trial = 10 * (i + 1) - 1
    if i == 0:  # 最初のセットではabs(past_best_score - current_best_score)が0になるため回避
        past_best_score = study.best_trial.value
        current_best_score = 0
