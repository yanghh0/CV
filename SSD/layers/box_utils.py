# -*- coding: utf-8 -*-
import torch


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:] / 2,
                     boxes[:, :2] + boxes[:, 2:] / 2), 1)


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2]) / 2,
                     boxes[:, 2:] - boxes[:, :2], 1)


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].

    后面我们代入得参数: box_a 包含所有的gt box，box_b包含所有的先验框
    一般情况下，先验框个数远大于真实框个数，即 A << B
    """
    A = box_a.size(0)   # A 是这副图像的真实框个数
    B = box_b.size(0)   # B = 8732

    # 每个真实框都要与所有的先验框计算 IOU
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]  # 交集的面积，形状为 (A,B)


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)

    # 计算每个真实框的面积
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    # 计算每个先验框的面积
    area_b = ((box_b[:, 2] - box_b[:, 0]) *
              (box_b[:, 3] - box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B], 返回交并比


# 输入包括IoU阈值、真实边框位置、预选框、方差、真实边框类别
# 输出每一个先验框的类别，保存在conf_t中；
# 输出每一个先验框与真实框的位置误差，保存在loc_t中
def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    
    # 注意这里truth是最大最小值形式的, 而prior是中心点与长宽形式
    # 求取每个真实框与每个预选框的IoU
    overlaps = jaccard(
        truths,
        point_form(priors)  # (cx, cy ,w, h) ---> (xmin, ymin, xmax, ymax)
    )

    # (Bipartite Matching) 双向匹配
    # 在维度1上取最大值，表示：找到和 gt box 最匹配的 prior box(anchor)，这是第一个方向的匹配
    # best_prior_overlap 形如: [[0.2500], [1.0000], [0.2500]]  [[0], [3], [5]]
    # best_prior_idx 的索引表示真实框的 id，元素表示对应 IOU 最大的先验框 id
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # 在维度0上取最大值，表示：找到和 anchor box 最匹配的 gtbox，这是第二个方向的匹配
    # best_truth_overlap 形如: [[0.2500, 0.4444, 0.2000, 1.0000, 0.2500, 0.2500]] [[0, 1, 1, 1, 2, 2]]
    # best_truth_idx 的索引表示先验框 id，元素表示对应 IOU 最大的真实框 id
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)

    # 下面几行就是去掉冗余维度
    best_prior_idx.squeeze_(1)        # 与gtbox最匹配的anchor索引值
    best_prior_overlap.squeeze_(1)    # 与gtbox最匹配的anchor交并比
    best_truth_idx.squeeze_(0)        # 与anchor最匹配的gtbox索引值
    best_truth_overlap.squeeze_(0)    # 与anchor最匹配的gtbox交并比

    # 以gt_box为准指定好与它自己最匹配的default box, 用于训练。
    # 将每一个truth对应的最佳box的overlap设置为2，第一个方向的匹配优先
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior

    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    # 每一个 prior 对应的真实框的位置，一个真实框可能对应好几个先验框
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]

    # 每一个 prior 对应的类别
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]

    # 如果一个 Prior Box 对应的最大 IoU 小于0.5，则视为负样本，背景类的位置误差不需要去拟合
    conf[best_truth_overlap < threshold] = 0  # label as background

    # 得到位置误差
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # matches:[xmin, ymin, xmax, ymax] default box: [cx, cy, w, h]
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    通过先验框和网络输出的位置误差得到真实框的位置
    """

    boxes = torch.cat((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    # (cx,cy,w,h) -> (xmin, ymin, xmax, ymax)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    # new是创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep

    # (xmin, ymin, xmax, ymax)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)

    # 将所有框的得分排序，选中最高分的 top_k 个框
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]       # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # 在 topK 个框中，选出最高分的框，遍历其余的框，如果和当前最高分框的IOU大于一定阈值，我们就将框删除
    # 从未处理的框中继续选一个得分最高的，重复上述过程
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view

        # load bboxes of next highest vals
        # topK 剩下的boxes的信息存储在xx，yy中
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        # store element-wise max with next highest score
        # 计算当前最大置信框与其他剩余框的交集区域坐标
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])

        w.resize_as_(xx2)
        h.resize_as_(yy2)

        w = xx2 - xx1
        h = yy2 - yy1

        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h

        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)

        union = (rem_areas - inter) + area[i]

        IoU = inter / union  # store result in iou

        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count
