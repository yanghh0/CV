# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from data import voc as cfg
from layers.box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label    # 背景类对应的label编号，默认0
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            targets (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        # loc_data : (batch_size, 8732, 4)
        # conf_data: (batch_size, 8732, 21)
        # priors   : (8732, 4)
        loc_data, conf_data, priors = predictions

        # num 即 batch_size
        num = loc_data.size(0)

        # 这一句感觉没啥用
        priors = priors[:loc_data.size(1), :]

        # num_priors = 8732
        num_priors = (priors.size(0))

        # num_classes = 21
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :-1].data   # shape: (num_objs, 4)
            labels = targets[idx][:, -1].data    # shape: (num_objs)
            defaults = priors.data               # shape: (8732, 4)
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        #===========================================================================
        # 计算所有正样本的定位损失,负样本不需要定位损失
        #===========================================================================

        # 计算正样本的数量
        pos = conf_t > 0                        # shape: (batch_size, 8732)
        num_pos = pos.sum(dim=1, keepdim=True)  # shape: (batch_size, 1)

        # 拓展为 Shape: [batch_size, 8732, 4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        # 正样本的预测位置误差 shape: (一个 batch 的正样本数量, 4)
        loc_p = loc_data[pos_idx].view(-1, 4)

        # 正样本的真实位置误差 shape: (一个 batch 的正样本数量, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

        # 所有正样本的定位损失
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        #===========================================================================
        # 对于类别损失，进行难样本挖掘，一张图片正负样本的比例控制为1:3
        #===========================================================================

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)  # shape: (batch_size * 8732, 21)

        # 只计算对应正确类别的置信度，并乘了一个负号，本来是找小的，现在变成找大的。
        # shape: (batch_size * 8732, 1)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)   # shape (batch_size, 8732)
        loss_c[pos] = 0                 # 过滤掉正样本

        _, loss_idx = loss_c.sort(1, descending=True)    # 将每个框正确类别的置信度降序排列
        _, idx_rank = loss_idx.sort(1)                   # 获取每个图片 8732 个框置信度(取了相反数的)排名
        num_pos = pos.long().sum(1, keepdim=True)        # shape: (batch_size, 1)

        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
        # 提取排名在前面的负样本，返回一个索引的 tensor
        neg = idx_rank < num_neg.expand_as(idx_rank) # shape (batch_size, 8732)

        #===========================================================================
        # 计算正负样本的类别损失
        #===========================================================================

        # Confidence Loss Including Positive and Negative Examples
        # 索引数组都扩展为[batch_size, 8732, 21]
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        # shape: (一个 batch 正样本数量 + 负样本数量, num_classes)
        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        # shape: (一个 batch 正样本数量 + 负样本数量)
        targets_weighted = conf_t[(pos + neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        N = num_pos.data.sum().type('torch.cuda.FloatTensor')
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
