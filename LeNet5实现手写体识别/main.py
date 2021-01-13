#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import cv2
import torch
import gzip, struct
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch.utils import data
import torch.optim as optim

BATCH_SIZE = 50
EPOCHS = 10
USE_GPU = True
USE_STANDARD_MINIST = True
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MyMnistDataset(data.Dataset):
    def __init__(self, file_path, is_train_set=True):
        self.file_path = file_path
        image_file_name = 'train-images-idx3-ubyte.gz' if is_train_set else 't10k-images-idx3-ubyte.gz'
        label_file_name = 'train-labels-idx1-ubyte.gz' if is_train_set else 't10k-labels-idx1-ubyte.gz'

        x, y = self._read(image_file_name, label_file_name)

        self.images = torch.from_numpy(x.astype(np.float32)).reshape(-1, 1, 28, 28)
        self.labels = torch.from_numpy(y.astype(np.int64))
        self.len = len(self.images)
        self.eight = self.FindEight()

    def __getitem__(self, index):
        if self.labels[index] == 0:    # 将数字 0 的图像和 8 叠加形成新的图像，该新图像的类别为 0
            self.images[index].data += self.eight.data
            # cv2.imshow("img", self.images[index][0].numpy())
            # cv2.waitKey()
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.len

    def _read(self, image, label):
        with gzip.open(os.path.join(self.file_path, image), 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            x = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(-1, rows, cols)

        with gzip.open(os.path.join(self.file_path, label), 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            y = np.frombuffer(flbl.read(), dtype=np.int8)

        return x, y

    def FindEight(self):
        """
        找到一个数字 8 对应的图像并保存
        """
        for i in range(self.len):
            if self.labels[i] == 8:
                return self.images[i]
        return None

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

        self.pooling = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # input: (bitch_size, 1, 28, 28)  output: (bitch_size, 6, 28, 28)
        x = self.conv1(x)
        x = f.relu(x)

        # input: (bitch_size, 6, 28, 28)  output: (bitch_size, 6, 14, 14)
        x = self.pooling(x)

        # input: (bitch_size, 6, 14, 14)  output: (bitch_size, 16, 10, 10)
        x = self.conv2(x)
        x = f.relu(x)

        # input: (bitch_size, 16, 10, 10) output: (bitch_size, 16, 5, 5)
        x = self.pooling(x)

        x = x.view(x.size(0), -1)  # # 16 * 5 * 5 = 400

        # input: (bitch_size, 400)  output: (bitch_size, 120)
        x = self.fc1(x)
        x = f.relu(x)

        # input: (bitch_size, 120)  output: (bitch_size, 84)
        x = self.fc2(x)
        x = f.relu(x)

        # input: (bitch_size, 84)  output: (bitch_size, 10)
        x = self.fc3(x)

        return x

def TrainModel(train_loader):
    running_loss = 0.0
    batch_num = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        if USE_GPU:
            inputs, target = inputs.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_num += 1

    print('loss: %.3f' % (running_loss / batch_num))

def TestModel(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            if USE_GPU:
                inputs, target = inputs.to(DEVICE), target.to(DEVICE)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    print('accuracy on test set: %.2f %% ' % (100 * correct / total))

    return correct / total

if __name__ == '__main__':
    # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),       # 把 (H,W,C) 的矩阵转为 (C,H,W)
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if USE_STANDARD_MINIST:
        train_set = datasets.MNIST('data', train=True, transform=transform, download=True)
    else:
        train_set = MyMnistDataset(os.path.join('data', 'MNIST', 'raw'), is_train_set=True)
    train_loader = data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    if USE_STANDARD_MINIST:
        test_set = datasets.MNIST('data', train=False, transform=transform, download=True)
    else:
        test_set = MyMnistDataset(os.path.join('data', 'MNIST', 'raw'), is_train_set=False)
    test_loader = data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = LeNet()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    if USE_GPU:
        model.to(DEVICE)

    acc_list = []
    for epoch in range(EPOCHS):
        TrainModel(train_loader)
        acc = TestModel(test_loader)
        acc_list.append(acc)

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
