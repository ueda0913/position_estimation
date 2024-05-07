import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from definitions.mydataset import MyGPUdataset
from definitions.net import select_net
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

## change area
epoch = 4000
node = 8
batch_size = 16
model_name = "vit_b16"  # vgg19_bn or mobilenet_v2 or resnet_152 or vit_b16
load_dir_name = "2024-01-08-12"
n_middle = 256
file_name = "no_title"
criterion = torch.nn.CrossEntropyLoss()


def save_confusion_matrix(
    cm,
    classes,
    normalize=False,
    title=f"Confusion matrix",
    cmap=plt.cm.Blues,
    save_path=None,
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    #     print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    if save_path == None:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches="tight")


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
# train_for_cmls(cur_dir, epoch - 1, node, classes, net, criterion, test_loader, device)
n_val_acc = 0
val_loss = 0
n_test = 0
y_preds = []  # for confusion_matrix
y_tests = []
y_outputs = []
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
    y_preds.extend(predicted_test.tolist())
    y_tests.extend(labels_test.tolist())
    y_outputs.extend(outputs_test.tolist())
    val_loss += loss_test.item() * test_batch_size
    n_val_acc += (predicted_test == labels_test).sum().item()
normalized_cm_dir_path = os.path.join(cur_dir, "images/normalized_confusion_matrix")
if not (os.path.exists(normalized_cm_dir_path)) or os.path.isfile(
    normalized_cm_dir_path
):
    os.makedirs(normalized_cm_dir_path)
confusion_mtx = confusion_matrix(y_tests, y_preds)
print("Saving confusion matrix...")
save_confusion_matrix(
    confusion_mtx,
    classes=classes,
    normalize=True,
    title=f"Normalized Confusion Matrix at {epoch+1:d}epoch (node{node})",
    cmap=plt.cm.Reds,
    save_path=os.path.join(
        cur_dir,
        f"images/normalized_confusion_matrix/{file_name}-node{node}.png",
    ),
)
