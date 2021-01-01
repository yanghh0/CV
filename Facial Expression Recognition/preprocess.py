#!/usr/bin/env python3
# -*- coding:utf8 -*-

import os
import cv2
import numpy as np
import pandas as pd

df = pd.read_csv(os.path.join('dataset', 'train.csv'))
x = df[['feature']]
y = df[['label']]
x.to_csv(os.path.join('dataset', 'data.csv'), index=False, header=False)
y.to_csv(os.path.join('dataset', 'label.csv'), index=False, header=False)

data = np.loadtxt(os.path.join('dataset', 'data.csv'))
if not os.path.isdir(os.path.join('dataset', 'face')):
    os.mkdir(os.path.join('dataset', 'face'))
if not os.path.isdir(os.path.join('dataset', 'test')):
    os.mkdir(os.path.join('dataset', 'test'))

"""
共有28709张图片，取前24000张图片作为训练集，其他图片作为验证集。
"""
for i in range(data.shape[0]):
    faceImg = data[i, :].reshape((48, 48))
    if i <= 24000:
        cv2.imwrite(os.path.join('dataset', 'face', '{}.jpg').format(i), faceImg)
    else:
        cv2.imwrite(os.path.join('dataset', 'test', '{}.jpg').format(i), faceImg)