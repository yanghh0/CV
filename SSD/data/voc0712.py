"""VOC Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os.path as osp
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

# VOC的数据类别有 20 种，always index 0
VOC_CLASSES = (
    'aeroplane', 
    'bicycle', 
    'bird', 
    'boat',
    'bottle', 
    'bus', 
    'car', 
    'cat', 
    'chair',
    'cow', 
    'diningtable', 
    'dog', 
    'horse',
    'motorbike', 
    'person', 
    'pottedplant',
    'sheep', 
    'sofa', 
    'train', 
    'tvmonitor')

# 需要将数据集 VOCdevkit 文件夹放到 SSD 的同级目录
VOC_ROOT = osp.join("..", "VOCdevkit")


class VOCAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not (default: False)
        height (int): height
        width (int): width

    解释：
        将VOC的annotation中的bbox坐标转化为归一化的值；
        将类别转化为用索引来表示的字典形式；
        参数列表：
            class_to_ind: 类别的索引字典。
            keep_difficult: 是否保留difficult=1的物体。
    """

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]

        解释：
        参数列表：
            target: xml被读取的一个ET.Element对象
            width: 图片宽度
            height: 图片高度
        返回值：
            一个list，list中的每个元素是[bbox coords, class name]

        """
        res = []
        # 遍历 xml 文件中所有的 object
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width，即归一化，x/w，y/h
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            # 提取类别名称对应的 index
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder，即 VOC_ROOT
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the input image
        target_transform (callable, optional): transformation to perform on the target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load (default: 'VOC2007')

    解释：
        根据VOCAnnotationTransform()和VOC数据集的文件结构，读取图片，bbox和lable，构建VOC数据集。
    """

    def __init__(self, root,
                 image_sets=[('2007', 'trainval'),],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = osp.join('%s', 'Annotations', '%s.xml')
        self._imgpath = osp.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()

        for (year, name) in image_sets:
            rootpath = osp.join(self.root, 'VOC' + year)
            for line in open(osp.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, channels = img.shape

        # 对xml执行VOCAnnotationTransform()操作
        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            # bgr to rgb
            img = img[:, :, (2, 1, 0)]
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        # return torch.from_numpy(img), target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)
