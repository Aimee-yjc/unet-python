# -*- coding: UTF-8 -*-

"""
@Project ：unet 图像分割
@File    ：dataset.py
@IDE     ：PyCharm
@Author  ：Aimee
@Date    ：2022/4/19
"""

import torch
import argparse
from torch.utils.data import DataLoader
from torch import optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
import matplotlib
import matplotlib.pyplot as plt
from dataset import tensor_to_PIL
from torch.utils.tensorboard import SummaryWriter

# 是否使用cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 把多个步骤整合到一起, channel=（channel-mean）/std, 因为是分别对三个通道处理
x_transforms = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # ->[-1,1]
])

# mask只需要转换为tensor
y_transforms = transforms.ToTensor()

# 参数解析器,用来解析从终端读取的命令
parse = argparse.ArgumentParser()


def train_model(model, criterion, optimizer, dataload, num_epochs=20):
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            labels = labels[:,0,:,:]
            print(labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
            # # 可视化
            # writer = SummaryWriter(log_dir='logs', flush_secs=60)
            # writer.add_scalar('Train_loss', loss, epoch)
        torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    return model


# 训练模型
def train():
    model = Unet(3, 1).to(device)
    batch_size = args.batch_size
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    # print(dataloaders)
    train_model(model, criterion, optimizer, dataloaders)



# 显示模型的输出结果
def test():
    model = Unet(3, 1)
    model.load_state_dict(torch.load(args.ckp, map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()

    # plt.ion()
    with torch.no_grad():
        for x, true_y in dataloaders:
            y = model(x)
            img_y = torch.squeeze(y).numpy()

            x = tensor_to_PIL(x)
            true_y = tensor_to_PIL(true_y)
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(x)
            plt.subplot(1, 3, 2)
            plt.imshow(img_y)
            plt.subplot(1, 3, 3)
            plt.imshow(true_y)
            plt.pause(0.1)
        plt.show()


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    # parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--ckp", type=str, help="the path of model weight file")
    args = parse.parse_args()

    # train
    # train()

    # test()
    args.ckp = "weights_19.pth"
    test()