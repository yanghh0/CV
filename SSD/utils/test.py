
"""
该文件用来测试augmentations.py
"""

import os
import cv2
from matplotlib import pyplot as plt
import utils.augmentations as aug

imgFile = os.path.join("..", "data", "example.jpg")

def image_compare(filePath, transform):
    """
    filePath (string): 图片文件路径
    transform (object): 可传参的对象
    """
    # 读取原始图片
    img = cv2.imread(filePath)
    plt.figure(figsize=(16,14))
    plt.subplot(2,1,1)
    plt.imshow(img)
    # 调用这个实列化后的对象
    image, _, _ = transform(img)
    # 读取第二张图片
    plt.subplot(2,1,2)
    plt.imshow(image)
    plt.show()
    # print(image)  # image 一个三维的张量

randomsaturation = aug.ConvertColor()
image_compare(imgFile, randomsaturation)

trans = aug.RandomLightingNoise()
image_compare(imgFile, trans)

trans = aug.RandomBrightness()
image_compare(imgFile, trans)