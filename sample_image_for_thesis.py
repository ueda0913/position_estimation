import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

data_dir = "../data-raid/data/position_estimation_dataset"
train_dir = os.path.join(data_dir, "train")
img_for_thesis_dir = os.path.join(data_dir, "samples")
n_node = 12
img_num = 3


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(
        (
            (img_width - crop_width) // 2,
            (img_height - crop_height) // 2,
            (img_width + crop_width) // 2,
            (img_height + crop_height) // 2,
        )
    )


if not (os.path.exists(img_for_thesis_dir)):
    os.makedirs(img_for_thesis_dir)

plt.figure(figsize=(12, 4))
plt.subplots_adjust(wspace=0.05)
cur_num = 0
for i in range(n_node):
    label_path = os.path.join(train_dir, str(i))
    img_path_list = os.listdir(label_path)
    for image_path in img_path_list:
        if cur_num == 3:
            plt.clf()
            cur_num = 0
            break
        cur_num += 1
        full_image_path = os.path.join(label_path, image_path)
        image = Image.open(full_image_path)
        ax = plt.subplot(1, 3, cur_num)
        im = image.rotate(-90)
        im = crop_center(im, min(im.size) - 1, min(im.size) - 1)
        plt.imshow(im)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if cur_num == 3:
            plt.savefig(
                os.path.join(img_for_thesis_dir, f"sample_l{i}.png"),
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
