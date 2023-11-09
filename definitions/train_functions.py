import itertools
import os
import random
import warnings

warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.io as io
import torchvision.transforms as transforms
from definitions.mydataset import *
from definitions.visualize import *
from torch.utils.data import DataLoader

# time_index = datetime.now().strftime("%Y-%m-%d-%H")

# デフォルトフォントサイズ変更
# plt.rcParams['font.size'] = 14
# # デフォルトグラフサイズ変更
# plt.rcParams['figure.figsize'] = (6,6)
# # デフォルトで方眼表示ON
# plt.rcParams['axes.grid'] = True
# np.set_printoptions(suppress=True, precision=5)


def pre_train(
    nets,
    train_loaders,
    test_loader,
    optimizers,
    criterion,
    num_epoch,
    device,
    cur_dir,
    historys,
    pre_train_historys,
    schedulers,
):
    for epoch in range(num_epoch):
        for n in range(len(train_loaders)):
            nets[n].train()
            data_train_num = 0
            n_train_acc, n_val_acc = 0, 0
            train_loss, val_loss = 0, 0
            for data in train_loaders[n]:
                # get the inputs; data is a list of [x_train, y_train]
                x_train, y_train = data
                batch_size = len(y_train)
                if x_train.device == "cpu":
                    x_train = x_train.to(device)
                    y_train = y_train.to(device)
                optimizers[n].zero_grad()
                y_output = nets[n](x_train)
                loss = criterion(y_output, y_train)
                loss.backward()
                optimizers[n].step()
                predicted = torch.max(y_output, 1)[1]
                n_train_acc += (predicted == y_train).sum().item()
                data_train_num += batch_size
                train_loss += loss.item() * batch_size

            nets[n].eval()
            data_test_num = 0
            for tdata in test_loader:
                x_test, y_test = tdata
                batch_size = len(y_test)
                if x_test.device == "cpu":
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                y_output_t = nets[n](x_test)
                loss_t = criterion(y_output_t, y_test)
                predicted_t = torch.max(y_output_t, 1)[1]
                n_val_acc += (predicted_t == y_test).sum().item()
                data_test_num += batch_size
                val_loss += loss_t.item() * batch_size

            train_acc = n_train_acc / data_train_num
            val_acc = n_val_acc / data_test_num
            avg_train_loss = train_loss / data_train_num
            avg_val_loss = val_loss / data_test_num

            if epoch % 5 == 4:
                print(
                    f"Pre-self training: [{n}th-node, {epoch + 1}th-epoch] train_acc: {train_acc:.5f}, val_acc: {val_acc:.5f}"
                )
            if epoch % (num_epoch // 2) == (num_epoch // 2) - 1:
                with open(os.path.join(cur_dir, "log.txt"), "a") as f:
                    f.write(
                        f"Pre-self training: [{n}th-node, {epoch + 1}th-epoch] train_acc: {train_acc:.5f}, val_acc: {val_acc:.5f}\n"
                    )
            item = np.array([0, avg_train_loss, train_acc, avg_val_loss, val_acc])
            item_p = np.array(
                [epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc]
            )
            pre_train_historys[n] = np.vstack((pre_train_historys[n], item_p))
            if epoch == num_epoch - 1:
                historys[n] = np.vstack((historys[n], item))
            if schedulers != None:
                schedulers[n].step()

    for n in range(len(train_loaders)):
        torch.save(
            nets[n].state_dict(), os.path.join(cur_dir, f"params/Pre-train-node{n}.pth")
        )


def fit(
    net,
    optimizer,
    criterion,
    train_loader,
    test_loader,
    device,
    history,
    cur_epoch,
    cur_node,
    evaluate_only=False,
):
    n_train_acc, n_val_acc = 0, 0
    train_loss, val_loss = 0, 0
    n_train, n_test = 0, 0

    if not evaluate_only:
        net.train()
        for inputs, labels in train_loader:
            train_batch_size = len(labels)
            n_train += train_batch_size
            if inputs.device == "cpu":
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            predicted = torch.max(outputs, 1)[1]
            train_loss += loss.item() * train_batch_size
            n_train_acc += (predicted == labels).sum().item()

    net.eval()
    for inputs_test, labels_test in test_loader:
        test_batch_size = len(labels_test)
        n_test += test_batch_size
        if inputs_test.device == "cpu":
            inputs_test = inputs_test.to(device)
            labels_test = labels_test.to(device)

        outputs_test = net(inputs_test)
        loss_test = criterion(outputs_test, labels_test)

        predicted_test = torch.max(outputs_test, 1)[1]
        val_loss += loss_test.item() * test_batch_size
        n_val_acc += (predicted_test == labels_test).sum().item()

    if not evaluate_only:
        train_acc = n_train_acc / n_train
        avg_train_loss = train_loss / n_train
    val_acc = n_val_acc / n_test
    avg_val_loss = val_loss / n_test
    print(
        f"Epoch [{cur_epoch+1}], Node [{cur_node}], loss: {avg_train_loss:.5f} acc: {train_acc:.5f} val_loss: {avg_val_loss:.5f} val_acc: {val_acc:.5f}"
    )
    item = np.array([cur_epoch + 1, avg_train_loss, train_acc, avg_val_loss, val_acc])
    history = np.vstack((history, item))
    return history


def torch_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def show_dataset_contents(
    data_path, classes, result_path
):  # classes: ("label1", "label2", ...), print dataset divided by classes
    num_test = [0 for i in range(len(classes))]
    num_train = [0 for i in range(len(classes))]
    for i, label in enumerate(classes):
        class_path_train = os.path.join(data_path, "train", str(i))
        files_and_dirs_in_container_train = os.listdir(class_path_train)
        files_list_train = [
            d
            for d in files_and_dirs_in_container_train
            if os.path.isfile(os.path.join(class_path_train, d))
        ]
        num_train[i] = len(files_list_train)

        class_path_test = os.path.join(data_path, "val", str(i))
        files_and_dirs_in_container_test = os.listdir(class_path_test)
        files_list_test = [
            d
            for d in files_and_dirs_in_container_test
            if os.path.isfile(os.path.join(class_path_test, d))
        ]
        num_test[i] = len(files_list_test)

    with open(os.path.join(result_path, "log.txt"), "w") as f:
        for i in range(len(num_test)):
            # print(f'label: {classes[i]} train_data: {num_train[i]} test_data: {num_test[i]}')
            f.write(
                f"label: {classes[i]} train_data: {num_train[i]} test_data: {num_test[i]}\n"
            )


def search_mean_and_std(datapath):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(datapath, transform=transform)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = len(dataset)

    for i in range(total_samples):
        image, _ = dataset[i]
        # print(image.mean())
        mean += image.mean(dim=(1, 2))
        std += image.std(dim=(1, 2))

    mean /= total_samples
    std /= total_samples
    return mean, std


def make_trainloader(subset, means, stds, batch_size, g):
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
    return trainloader
