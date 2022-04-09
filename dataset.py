# -*- coding: UTF-8 -*-

"""
@Project ：unet 图像分割
@File    ：dataset.py
@IDE     ：PyCharm 
@Author  ：Aimee
@Date    ：2022/4/19
"""

import torch.utils.data as data
import PIL.Image as Image
import os
from torchvision.transforms import transforms


def make_dataset(root):
    imgs = []
    n = len(os.listdir(root)) // 2
    for i in range(n):
        img = os.path.join(root, "%03d.png" % i)
        mask = os.path.join(root, "%03d_mask.png" % i)
        imgs.append((img, mask))
    return imgs


class LiverDataset(data.Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path).convert('RGB')
        resize = transforms.Resize([512, 512])
        img_x = resize(img_x)
        img_y = Image.open(y_path).convert('RGB')
        img_y = resize(img_y)
        # img_y = img_y[:,0,:,:]
        if self.transform is not None:
            img_x = self.transform(img_x)
            # img_y = self.transform(img_y)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image