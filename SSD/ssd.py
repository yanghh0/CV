import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
from data import voc
import os


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = voc
        self.priorbox = PriorBox(self.cfg)
        self.priors = self.priorbox.forward()
        self.size = size

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """

        # sources保存特征图，loc与conf保存所有PriorBox的位置与类别预测特征
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        # 对输入图像卷积到conv4_3，将特征添加到sources中
        for k in range(23):
            x = self.vgg[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # apply vgg up to fc7
        # 继续卷积到conv7，将特征添加到sources中
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        # 继续利用额外的卷积层计算，并将特征添加到sources中
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        # 对sources中的特征图利用类别与位置网络进行卷积计算，并保存到loc与conf中
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())   # (batch_size, c, h, w) -> (batch_size, h, w, c)
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())  # (batch_size, c, h, w) -> (batch_size, h, w, c)

        # 变成 [batch_size，8732 * 4]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 变成 [batch_size，8732 * 21]
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                                     # loc preds
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)),      # conf preds
                self.priors.type(type(x.data))                                    # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),                   # shape: (batch_size, 8732, 4)
                conf.view(conf.size(0), -1, self.num_classes),  # shape: (batch_size, 8732, 21)
                self.priors                                     # shape: (8732, 4)
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
# 搭建vgg16基础网络的函数
# cfg: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i

    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v

    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    # dilation设置带孔/空洞卷积
    conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


# [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
# 这里没有设置relu层，留到整体构造网络的时候再设置
def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels=in_channels, 
                                     out_channels=cfg[k + 1], 
                                     kernel_size=(1, 3)[flag], 
                                     stride=2, 
                                     padding=1)]
            else:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers


# cfg = [4, 6, 6, 6, 4, 4]
# 每个先验框需要输出: 位置[dx,dy,dw,dh] + 21个类别的置信度 = 4 + num_classes = 25
# 先验框个数总共为: 38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4 = 8732
def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]

    # Conv4_3(38*38), Conv7(19*19)
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(in_channels=vgg[v].out_channels, 
                                 out_channels=cfg[k] * 4,             # 4 表示 [dx,dy,dw,dh]
                                 kernel_size=3, 
                                 padding=1)]
        conf_layers += [nn.Conv2d(in_channels=vgg[v].out_channels, 
                                  out_channels=cfg[k] * num_classes,  # num_classes 是 21 个类别
                                  kernel_size=3, 
                                  padding=1)]

    # Conv8_2(10*10), Conv9_2(5*5), Conv10_2(3*3), Conv11_2(1*1)
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(in_channels=v.out_channels, 
                                 out_channels=cfg[k] * 4, 
                                 kernel_size=3, 
                                 padding=1)]
        conf_layers += [nn.Conv2d(in_channels=v.out_channels, 
                                  out_channels=cfg[k] * num_classes, 
                                  kernel_size=3, 
                                  padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


# 这里的base为VGG-16前13个卷积层构造，M代表maxpooling，C代表ceil_mode为True
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512],
    '512': [],
}

# 额外部分的卷积通道数，S代表了步长为2，其余卷积层默认步长为1
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}

# 每个特征图上一个点对应的PriorBox数量
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return

    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return

    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
