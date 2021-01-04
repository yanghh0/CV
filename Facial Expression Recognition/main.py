#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import cv2
import torch
import numpy as np
from torch.utils import data
import torch.nn.functional as f
import torch.optim as optim

class FaceDataset(data.Dataset):
    def __init__(self, filePath, start, end):
        y = np.loadtxt(os.path.join('dataset', 'label.csv'), dtype=np.float32)
        self.filePath = filePath
        self.start = start
        self.end = end
        self.yDatas = torch.from_numpy(y)[start:end]
        self.yDatas = self.yDatas.type('torch.LongTensor')
        self.len = self.yDatas.shape[0]

    def __getitem__(self, index):
        img = cv2.imread(os.path.join('dataset', self.filePath, '{}.jpg'.format(index + self.start)))
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)           # 转为单通道灰度图
        grayImgNormalized = grayImg.reshape(1, 48, 48) / 255.0    # 转为标准正态分布
        xData = torch.from_numpy(grayImgNormalized)
        xData = xData.type('torch.FloatTensor')
        return xData, self.yDatas[index]

    def __len__(self):
        return self.len

class Inception(torch.nn.Module):
    def __init__(self, inChannels):
        super(Inception, self).__init__()
        self.branchPool = torch.nn.Conv2d(in_channels=inChannels, out_channels=24, kernel_size=1)

        self.branch1x1 = torch.nn.Conv2d(in_channels=inChannels, out_channels=16, kernel_size=1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels=inChannels, out_channels=16, kernel_size=1)
        self.branch5x5_2 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels=inChannels, out_channels=16, kernel_size=1)
        self.branch3x3_2 = torch.nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)

    def forward(self, x):
        branchPool = f.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branchPool = self.branchPool(branchPool)

        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        outputs = [branch1x1, branch5x5, branch3x3, branchPool]
        return torch.cat(outputs, dim=1)  # b,c,w,h  c 对应的是 dim=1 88 = 24x3 + 16 输出一共 88 通道

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=30, kernel_size=5, padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=88, out_channels=20, kernel_size=5, padding=2)  # 88 = 24x3 + 16

        self.incep1 = Inception(inChannels=30)  # 与conv1 中的 30 对应
        self.incep2 = Inception(inChannels=20)  # 与conv2 中的 20 对应

        self.pooling = torch.nn.MaxPool2d(kernel_size=2)
        self.fc = torch.nn.Linear(in_features=12672, out_features=7)

    def forward(self, x):
        # input: (bitch_size, 1, 48, 48)  output: (bitch_size, 30, 48, 48)
        x = self.conv1(x)

        # input: (bitch_size, 30, 48, 48)  output: (bitch_size, 30, 24, 24)
        x = self.pooling(x)
        x = f.relu(x)

        # input: (bitch_size, 30, 24, 24)  output: (bitch_size, 88, 24, 24)
        x = self.incep1(x)

        # input: (bitch_size, 88, 24, 24)  output: (bitch_size, 20, 24, 24)
        x = self.conv2(x)

        # input: (bitch_size, 20, 24, 24)  output: (bitch_size, 20, 12, 12)
        x = self.pooling(x)
        x = f.relu(x)

        # input: (bitch_size, 20, 12, 12)  output: (bitch_size, 88, 12, 12)
        x = self.incep2(x)

        x = x.view(x.size(0), -1)  # 88 * 12 * 12 = 12672
        x = self.fc(x)
        return x

def TrainModel(trainLoader, testLoader, useGPU=False):
    model = Model()
    if useGPU:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    for epoch in range(10):
        runningLoss = 0.0
        for batchIdx, data in enumerate(trainLoader, 0):
            inputs, target = data
            if useGPU:
                inputs, target = inputs.to(device), target.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            if batchIdx % 300 == 299:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, batchIdx + 1, runningLoss / 300))
                runningLoss = 0.0

        correct, total = 0, 0
        with torch.no_grad():
            for data in testLoader:
                inputs, target = data
                if useGPU:
                    inputs, target = inputs.to(device), target.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, dim=1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        print('accuracy on test set: %d %% ' % (100 * correct / total))
                
    torch.save(model, 'model.pkl')

if __name__ == '__main__':
    trainSet = FaceDataset('face', 0, 24000 + 1)
    trainLoader = data.DataLoader(dataset=trainSet, batch_size=32, shuffle=True, num_workers=2)
    testSet = FaceDataset('test', 24001, 28708 + 1)
    testLoader = data.DataLoader(dataset=testSet, batch_size=32, shuffle=True, num_workers=2)
    TrainModel(trainLoader, testLoader, useGPU=True)