import itertools
import os
import statistics

# import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
from sklearn.metrics import confusion_matrix


def evaluate_history(historys, cur_dir):
    # numpy_array = np.linspace(0, 0.9, len(historys) * 2)
    # color_list = [(x, x, x) for x in numpy_array]
    color_list = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#b5bdab",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
        "#c49c94",
        "#f7b6d2",
        "#c7c7c7",
        "#dbdb8d",
        "#9edae5",
        "#adadad",
        "#ff7f0e",
        "#1f77b4",
        "#8c564b",
    ]
    with open(os.path.join(cur_dir, "log.txt"), "a") as f:
        f.write(f"{len(historys[0])}epochまでの学習\n")
        for i in range(len(historys)):
            f.write(
                f"初期状態(node{i}): 損失: {historys[i][0, 3]:.5f} 精度: {historys[i][0, 4]:.5f}\n"
            )
            f.write(
                f"最終状態(node{i}): 損失: {historys[i][-1, 3]:.5f} 精度: {historys[i][-1, 4]:.5f}\n"
            )
    img_dir_path = os.path.join(cur_dir, "images")
    if not (os.path.exists(img_dir_path)) or os.path.isfile(img_dir_path):
        os.makedirs(img_dir_path)
    num_epochs = len(historys[0])
    unit = num_epochs / 10

    plt.figure(figsize=(9, 8))
    for i in range(len(historys)):
        plt.plot(
            historys[i][:, 0],
            historys[i][:, 1],
            label=f"node{i}(training)",
            linewidth=0.5,
            color=color_list[i * 2],
        )
        plt.plot(
            historys[i][:, 0],
            historys[i][:, 3],
            label=f"node{i}(validation)",
            linewidth=0.5,
            color=color_list[i * 2 + 1],
        )
    plt.xticks(np.arange(0, num_epochs, unit))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve (loss)")
    plt.legend(ncol=2)
    plt.savefig(os.path.join(cur_dir, "images/loss.png"))

    plt.figure(figsize=(9, 8))
    for i in range(len(historys)):
        plt.plot(
            historys[i][:, 0],
            historys[i][:, 2],
            label=f"node{i}(training)",
            linewidth=0.5,
            color=color_list[i * 2],
        )
        plt.plot(
            historys[i][:, 0],
            historys[i][:, 4],
            label=f"node{i}(validation)",
            linewidth=0.5,
            color=color_list[i * 2 + 1],
        )
    plt.xticks(np.arange(0, num_epochs + 1, unit))
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Learning curve (accuracy)")
    plt.legend(ncol=2)
    plt.savefig(os.path.join(cur_dir, "images/acc.png"))


def show_image_labels(loader, classes, net, device, loaded_epoch, cur_dir, all_images):
    if all_images:
        plt.figure(figsize=(30, 80))
        row_num = (
            len(loader.dataset) // 10
            if len(loader.dataset) % 10 == 0
            else len(loader.dataset) // 10 + 1
        )
    else:
        plt.figure(figsize=(20, 15))
        row_num = (
            int(len(loader.dataset) * 0.3) // 10
            if len(loader.dataset) * 0.3 % 10 == 0
            else int(len(loader.dataset) * 0.3) // 10 + 1
        )
    index = 0
    wrong_data_num = 0
    for images, labels in loader:
        n_size = len(images)
        if net is not None:
            inputs = images.to(device)
            labels = labels.to(device)
            net.eval()
            outputs = net(inputs)
            predicted = torch.max(outputs, 1)[1]
            images = images.to("cpu")
        else:
            images = images.to("cpu")
            labels = labels.to("cpu")

        for i in range(n_size):
            label_name = classes[labels[i]]
            predicted_name = classes[predicted[i]]
            if all_images:
                ax = plt.subplot(row_num, 10, index * n_size + i + 1)
            elif label_name != predicted_name:
                ax = plt.subplot(row_num, 10, wrong_data_num + 1)
            else:
                continue
            if net is not None:
                if label_name == predicted_name:
                    c = "k"
                else:
                    c = "b"
                    wrong_data_num += 1
                ax.set_title(label_name + ":" + predicted_name, c=c, fontsize=20)
            else:
                ax.set_title(label_name, fontsize=20)

            image_np = images[i].numpy().copy()
            img = np.transpose(image_np, (1, 2, 0))
            img = (img + 1) / 2
            plt.imshow(img)
            ax.set_axis_off()
        index += 1
    plt.savefig(
        os.path.join(cur_dir, f"images/sample_images/si-epoch-{loaded_epoch}.png"),
        bbox_inches="tight",
        pad_inches=0,
    )


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
    plt.title(title)
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


def make_latent_space(y_tests, y_outputs, epoch, ls_path, cur_node):
    reducer = umap.UMAP(random_state=42)
    reducer.fit(y_outputs)
    embedding = reducer.transform(y_outputs)

    # plot
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_tests, cmap="Spectral", s=5)
    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(
        f"UMAP projection of the output features @ epoch={epoch:d}, node={cur_node}",
        fontsize=12,
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(ls_path, bbox_inches="tight")

    # plot
    plt.clf()
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y_tests, cmap="Spectral", s=5)
    plt.gca().set_aspect("equal", "datalim")
    plt.colorbar(boundaries=np.arange(11) - 0.5).set_ticks(np.arange(10))
    plt.title(
        f"UMAP projection of features @ epoch={epoch:d}, node={cur_node}", fontsize=12
    )
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig(ls_path, bbox_inches="tight")


def calc_res_mean_and_std(histories):  # 後ろ10epochの結果の平均と分散を求める
    raw_data = []
    for i in range(len(histories)):
        for j in range(10):  # 何このデータを使って求めるか
            raw_data.append(histories[i][-j - 1][4])
    mean = statistics.mean(raw_data)
    std = statistics.stdev(raw_data)
    max_acc = max(raw_data)
    min_acc = min(raw_data)
    return mean, std, max_acc, min_acc


def train_for_cmls(cur_dir, epoch, n, classes, net, criterion, test_loader, device):
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
        # outputs_test2 = tmp_net(inputs_test)
        loss_test = criterion(outputs_test, labels_test)

        predicted_test = torch.max(outputs_test, 1)[1]
        y_preds.extend(predicted_test.tolist())
        y_tests.extend(labels_test.tolist())
        y_outputs.extend(outputs_test.tolist())
        # z_outputs.extend(outputs_test2.tolist())
        val_loss += loss_test.item() * test_batch_size
        n_val_acc += (predicted_test == labels_test).sum().item()

    # make confusion matrix
    # cm_dir_path = os.path.join(cur_dir, "images/confusion_matrix")
    # if not (os.path.exists(cm_dir_path)) or os.path.isfile(cm_dir_path):
    #     os.makedirs(cm_dir_path)
    normalized_cm_dir_path = os.path.join(cur_dir, "images/normalized_confusion_matrix")
    if not (os.path.exists(normalized_cm_dir_path)) or os.path.isfile(
        normalized_cm_dir_path
    ):
        os.makedirs(normalized_cm_dir_path)
    confusion_mtx = confusion_matrix(y_tests, y_preds)
    # save_confusion_matrix(
    #     confusion_mtx,
    #     classes=classes,
    #     normalize=False,
    #     title=f"Confusion Matrix at {epoch+1:d}epoch (node{n})",
    #     cmap=plt.cm.Reds,
    #     save_path=os.path.join(
    #         cur_dir, f"images/confusion_matrix/cm-epoch-{epoch+1:04d}-node{n}.png"
    #     ),
    # )
    print("Saving confusion matrix...")
    save_confusion_matrix(
        confusion_mtx,
        classes=classes,
        normalize=True,
        title=f"Normalized Confusion Matrix at {epoch+1:d}epoch (node{n})",
        cmap=plt.cm.Reds,
        save_path=os.path.join(
            cur_dir,
            f"images/normalized_confusion_matrix/normalized-cm-epoch{epoch+1:04d}-node{n}.png",
        ),
    )

    # make ls
    ls_dir_path = os.path.join(cur_dir, "images/latent_space")
    if not (os.path.exists(ls_dir_path)) or os.path.isfile(ls_dir_path):
        os.makedirs(ls_dir_path)
    make_latent_space(
        y_tests,
        y_outputs,
        epoch + 1,
        os.path.join(ls_dir_path, f"ls-epoch{epoch+1:4d}-node{n}.png"),
        n,
    )
