#!/usr/bin/env python3
# -*- coding:utf8 -*-

import math
import cv2
import numpy as np

def Conv2D(kernel, img, padding, strides=(1, 1)):
    """
    单通道图像卷积
    """
    padImg = np.pad(img, padding, "edge")
    result = []
    for i in range(0, padImg.shape[0] - kernel.shape[0] + 1, strides[0]):
        result.append([])
        for j in range(0, padImg.shape[1] - kernel.shape[1] + 1, strides[1]):
            val = (kernel * padImg[i:i + kernel.shape[0], j:j + kernel.shape[1]]).sum()
            result[-1].append(val)
    return np.array(result)

def Map2Gray(data):
    """
    将数组中的元素映射到[0,255]区间
    """
    maxVal = np.max(data)
    minVal = np.min(data)
    interval = maxVal - minVal
    data = np.floor(255 / interval * (data - minVal))
    return data

def CalImgGrad(f):
    """
    用 5x5 sobel kernel 计算每个像素点的梯度
    """
    sobelx = np.array([
        [2, 1, 0, -1, -2],
        [2, 1, 0, -1, -2],
        [4, 2, 0, -2, -4],
        [2, 1, 0, -1, -2],
        [2, 1, 0, -1, -2]
    ])
    sobely = np.array([
        [2, 2, 4, 2, 2],
        [1, 1, 2, 1, 1],
        [0, 0, 0, 0, 0],
        [-1, -1, -2, -1, -1],
        [-2, -2, -4, -2, -2]
    ])

    fx = Conv2D(sobelx, f, ((2, 2), (2, 2)))
    fy = Conv2D(sobely, f, ((2, 2), (2, 2)))
    fx2 = fx * fx
    fy2 = fy * fy
    fxfy = fx * fy

    return fx2, fy2, fxfy

def CalRvals(fx2, fy2, fxfy):
    """
    计算每个像素对应的M矩阵和R值
    """
    W = 1.0 / 159 * np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2]
    ])
    k = 0.05

    fx2 = Conv2D(W, fx2, ((2, 2), (2, 2)))
    fy2 = Conv2D(W, fy2, ((2, 2), (2, 2)))
    fxfy = Conv2D(W, fxfy, ((2, 2), (2, 2)))

    result = []
    maxEValue = []   # 存储最大特征值
    minEValue = []   # 存储最小特征值
    for i in range(fx2.shape[0]):
        result.append([])
        maxEValue.append([])
        minEValue.append([])
        for j in range(fx2.shape[1]):
            M = np.array([
                [fx2[i, j], fxfy[i, j]],
                [fxfy[i, j], fy2[i, j]]
            ])
            a = np.linalg.det(M)
            b = np.trace(M)
            """
            M是实对称矩阵，按理必然存在实特征值，可能由于精度缺失，
            方程 x^2 - bx + a = 0 未必有解，所以 b ** 2 - 4 * a 取了个绝对值，
            这样算出来的值就不对。
            """
            delta = math.sqrt(math.fabs(b ** 2 - 4 * a))
            e1 = 0.5 * (b - delta)
            e2 = 0.5 * (b + delta)
            r = a - k * b
            result[-1].append(r)
            maxEValue[-1].append(max(e1, e2))
            minEValue[-1].append(min(e1, e2))

    g1 = Map2Gray(np.array(maxEValue))         # 最大特征值图
    g2 = Map2Gray(np.array(minEValue))         # 最小特征值图
    g3 = Map2Gray(np.array(result).copy())     # R图

    cv2.imshow('eMax', g1)
    cv2.waitKey(0)
    cv2.imshow('eMin', g2)
    cv2.waitKey(0)
    cv2.imshow('R', g3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(result), g1, g2, g3

def Sign(img, rVals):
    threshold = 25 * math.fabs(rVals.mean())
    judge = rVals >= threshold
    rectangle = (12, 12)    # 标记矩形大小
    thickness = 2           # 标记矩形边的厚度

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            up     = max(i - rectangle[0] // 2, 0)
            bottom = min(i + rectangle[0] // 2, img.shape[0] - 1)
            left   = max(j - rectangle[1] // 2, 0)
            right  = min(j + rectangle[1] // 2, img.shape[0] - 1)
            # 为了使矩形框不发生重叠需要进行非极值抑制
            isLocalExtreme = rVals[i,j] >= rVals[up:bottom + 1, left:right + 1]
            if judge[i, j] and isLocalExtreme.all():
                cv2.rectangle(img, (left, up), (right, bottom), (0, 0, 255), thickness)
    return img

def HarrisCornerDetection(img):
    grayImg = img.mean(axis=-1)  # 变成单通道灰度图像
    fx2, fy2, fxfy = CalImgGrad(grayImg)
    rVals, g1, g2, g3 = CalRvals(fx2, fy2, fxfy)
    finalImg = Sign(img, rVals)
    return finalImg, g1, g2, g3

# =============================================================================

def RecordVideo(fullFileName):
    fps = 25
    size = (640, 480)
    duration = 10  # 修改录制时长，单位s
    frameCount = duration * fps

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 打开摄像头
    videoWriter = cv2.VideoWriter(fullFileName, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)

    print("Begin to record %ss video" % str(duration))

    success, frame = cap.read()
    while success and frameCount > 0:
        videoWriter.write(frame)
        success, frame = cap.read()
        frameCount -= 1

    cap.release()
    print("End of recording")

def TestVideo():
    fullFileName = "video.avi"
    RecordVideo(fullFileName)

    triggerCount = 0
    capVideo = cv2.VideoCapture(fullFileName)
    fps = capVideo.get(cv2.CAP_PROP_FPS)
    frameInterval = int(1000 // fps)   # ms

    print()
    print("Begin to playback video")

    ret, videoFrame = capVideo.read()
    while ret:
        cv2.imshow('image', videoFrame)
        flag = cv2.waitKey(frameInterval)
        if flag == ord(' '):
            triggerCount += 1
            print("[%s]st trigger detection" % str(triggerCount))
            cv2.destroyAllWindows()

            finalImg, g1, g2, g3 = HarrisCornerDetection(videoFrame)
            cv2.imshow('image', finalImg)
            cv2.waitKey(0)

            cv2.imwrite("[%s]-e-max.jpg" % str(triggerCount), g1)
            cv2.imwrite("[%s]-e-min.jpg" % str(triggerCount), g2)
            cv2.imwrite("[%s]-R.jpg" % str(triggerCount), g3)
            cv2.imwrite("[%s]-final.jpg" % str(triggerCount), finalImg)
        ret, videoFrame = capVideo.read()

    capVideo.release()
    cv2.destroyAllWindows()
    print("End of playing")

def TestImage(fullFileName):
    img = cv2.imread(fullFileName)
    finalImg, g1, g2, g3 = HarrisCornerDetection(img)
    cv2.imshow('image', finalImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

TestVideo()
# TestImage("2.jpg")


