from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']       # [38, 19, 10, 5, 3, 1]
        self.min_sizes = cfg['min_sizes']             # [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes']             # [60, 111, 162, 213, 264, 315]
        self.steps = cfg['steps']                     # [8, 16, 32, 64, 100, 300] 各特征图相对于原图的缩放比例
        self.aspect_ratios = cfg['aspect_ratios']     # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    # 生成所有的PriorBox，需要每一个特征图的信息
    def forward(self):
        mean = []
        # 模拟每个特征图上的坐标
        for k, f in enumerate(self.feature_maps):   #  'feature_maps': [38, 19, 10, 5, 3, 1]
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]

                # 求每个box的中心坐标  将中心点坐标转化为 相对于 特征图的 相对坐标 （0，1）
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
