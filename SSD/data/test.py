"""
该文件用来测试voc0712文件中两个类 VOCDetection，VOCAnnotationTransform
"""

import cv2
import os.path as osp
import numpy as np
from torch.utils import data
from data.voc0712 import VOCDetection, VOC_CLASSES

VOC_ROOT = osp.join("..", "..", "VOCdevkit")

data_set = VOCDetection(VOC_ROOT)
data_loader = data.DataLoader(data_set, batch_size=1, num_workers=0, shuffle=True)

print('the data length is:', len(data_loader))

# 类别 to index
class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

# index to class，转化为类别名称
ind_to_class = {v: k for k, v in class_to_ind.items()}

# 加载数据
for datas in data_loader:
    img, target = datas
    img = img.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)  # 因为 batch_size=1，squeezequ去掉第0个维度

    h, w, c = img.shape
    print('h=%d, w=%d, c=%d' % (h, w, c))
    target = np.array(target)

    # 把bbox的坐标还原为原图的数值
    target[:, 0] *= float(w)
    target[:, 2] *= float(w)
    target[:, 1] *= float(h)
    target[:, 3] *= float(h)

    target = np.int0(target)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)   # RGB to BGR
    # 画出图中类别名称
    for i in range(target.shape[0]):
        # 画矩形框
        cv2.rectangle(img, (target[i, 0], target[i, 1]), (target[i, 2], target[i, 3]), (0, 0, 255), 2)
        # 标明类别名称
        img = cv2.putText(img, ind_to_class[target[i, 4]], (target[i, 0], target[i, 1] - 25),
                          cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 0), 1)
    # 显示
    cv2.imshow('imgs', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    break
