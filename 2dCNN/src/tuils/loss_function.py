import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val + \
               ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()

class BinaryEntropyLoss_weight(nn.Module):
    def __init__(self, weight=None, size_average=True, is_weight=True):
        super(BinaryEntropyLoss_weight, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_weight = is_weight
        self.class_num = np.array([[2.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        # self.class_num = np.power((1-self.class_num/50000), 2)
        # print(target.shape)

    def forward(self, input, target):

        self.weight = torch.cuda.FloatTensor(self.class_num.repeat(target.shape[0], axis=0))

        loss = F.binary_cross_entropy(input, target, self.weight, self.size_average)

        return loss

class FocalLoss_BCE(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss_BCE, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1)

        # pt = torch.sigmoid(input)
        pt = input
        pt = pt.view(-1)
        error = torch.abs(pt - target)
        log_error = torch.log(error)
        loss = -1 * (1-error)**self.gamma * log_error
        if self.size_average: return loss.mean()
        else: return loss.sum()
        
class BinaryEntropyLoss_weight_v2(nn.Module):
    def __init__(self, weight=None, size_average=True, is_weight=True):
        super(BinaryEntropyLoss_weight_v2, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_weight = is_weight

    def forward(self, input, target):

        if self.is_weight:
            total_pixel = target.numel()
            weights_list = []
            for i in range(2):
                if target[target==i].numel() == 0:
                    weights_list.append(0)
                else:
                    weights_list.append(target[target!=i].numel()/total_pixel)
            weights_list = np.clip(weights_list, 0.2, 0.8)
            self.weight = target.clone()
            self.weight[self.weight==0] = weights_list[0]
            self.weight[self.weight==1] = weights_list[1]
            # self.weight = torch.FloatTensor(self.weight).cuda()

        # loss_f = nn.BCELoss()
        loss = F.binary_cross_entropy(F.sigmoid(input), target, self.weight,self.size_average)
        # loss = F.binary_cross_entropy_with_logits(input, target, self.weight, reduce=False)
        # value, index= loss.topk(int(target.shape[1] * OHEM_percent), dim=1, largest=True, sorted=True)
        # return value.mean()
        return loss

class BinaryEntropyLoss_weight_v2_topk(nn.Module):
    def __init__(self, weight=None, size_average=True, is_weight=True):
        super(BinaryEntropyLoss_weight_v2_topk, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.is_weight = is_weight
        self.OHEM_percent = 0.1

    def forward(self, input, target):

        if self.is_weight:
            total_pixel = target.numel()
            # print(target.shape, total_pixel)
            weights_list = []
            for i in range(2):
                if target[target==i].numel() == 0:
                    weights_list.append(0)
                else:
                    weights_list.append(target[target!=i].numel()/total_pixel)
            weights_list = np.clip(weights_list, 0.2, 0.8)
            self.weight = target.clone()
            self.weight[self.weight==0] = weights_list[0]
            self.weight[self.weight==1] = weights_list[1]
            # self.weight = torch.FloatTensor(self.weight).cuda()

        # loss = F.binary_cross_entropy(F.sigmoid(input), target, self.weight,self.size_average)
        
        loss = F.binary_cross_entropy_with_logits(input, target, self.weight, reduce=False)
        loss = loss.view(loss.size(0), -1)
        value, index= loss.topk(int(target.shape[1] * target.shape[2] * self.OHEM_percent), dim=1, largest=True, sorted=True)
        return value.mean()
        # return loss

class SoftDiceLoss_binary_v2(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_binary_v2, self).__init__()

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.sigmoid(input).view(batch_size, -1)
        # print(target.shape)
        # print(target.view(-1))
        target = target.clone().view(batch_size, -1)

        input_b = 1 - input
        target_b = 1 - target

        inter_f = torch.sum(input * target, 1) + smooth
        union_f = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth
        score_f = torch.sum(2.0 * inter_f / union_f) / float(batch_size)

        inter_b = torch.sum(input_b * target_b, 1) + smooth
        union_b = torch.sum(input_b * input_b, 1) + torch.sum(target_b * target_b, 1) + smooth
        score_b = torch.sum(2.0 * inter_b / union_b) / float(batch_size)
        score = 1 - score_f - score_b
        # weight_f = score_b/(score_f + score_b)
        # weight_b = score_f/(score_f + score_b)
        # # score = 1.0 - torch.clamp(score_f, 0.0, 1.0 - 1e-7) - torch.clamp(score_b, 0.0, 1.0 - 1e-7)
        # score = 1 - weight_f * score_f - weight_b * score_b

        # inter_f = torch.sum(input * target, 1) + smooth
        # union_f = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth
        # score_f = inter_f / union_f

        # inter_b = torch.sum(input_b * target_b, 1) + smooth
        # union_b = torch.sum(input_b * input_b, 1) + torch.sum(target_b * target_b, 1) + smooth
        # score_b = inter_b / union_b    
        # score = 1 - torch.sum(score_f*score_b) / float(batch_size)

        return score
        
class SoftDiceLoss_binary_v3(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_binary_v3, self).__init__()

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.sigmoid(input).view(batch_size, -1)
        # print(target.shape)
        # print(target.view(-1))
        target = target.clone().view(batch_size, -1)

        input_b = 1 - input
        target_b = 1 - target

        inter_f = torch.sum(input * target, 1) + smooth
        union_f = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth
        score_f = torch.sum(2.0 * inter_f / union_f) / float(batch_size)

        inter_b = torch.sum(input_b * target_b, 1) + smooth
        union_b = torch.sum(input_b * input_b, 1) + torch.sum(target_b * target_b, 1) + smooth
        score_b = torch.sum(2.0 * inter_b / union_b) / float(batch_size)
        # score = 1 - score_f - score_b
        weight_f = score_b/(score_f + score_b)
        weight_b = score_f/(score_f + score_b)
        # score = 1.0 - torch.clamp(score_f, 0.0, 1.0 - 1e-7) - torch.clamp(score_b, 0.0, 1.0 - 1e-7)
        score = 1 - weight_f * score_f - weight_b * score_b

        # inter_f = torch.sum(input * target, 1) + smooth
        # union_f = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth
        # score_f = inter_f / union_f

        # inter_b = torch.sum(input_b * target_b, 1) + smooth
        # union_b = torch.sum(input_b * input_b, 1) + torch.sum(target_b * target_b, 1) + smooth
        # score_b = inter_b / union_b    
        # score = 1 - torch.sum(score_f*score_b) / float(batch_size)

        return score
        
class SoftDiceLoss_binary(nn.Module):
    def __init__(self):
        super(SoftDiceLoss_binary, self).__init__()

    def forward(self, input, target):
        smooth = 0.01
        batch_size = input.size(0)
        input = F.sigmoid(input).view(batch_size, -1)
        # print(target.shape)
        # print(target.view(-1))
        target = target.clone().view(batch_size, -1)

        inter = torch.sum(input * target, 1) + smooth
        union = torch.sum(input * input, 1) + torch.sum(target * target, 1) + smooth

        score = torch.sum(2.0 * inter / union) / float(batch_size)
        score = 1.0 - torch.clamp(score, 0.0, 1.0 - 1e-7)

        return score
        
class SoftDiceLoss(nn.Module):
    def __init__(self):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    # loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss

def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels

def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

class SymmetricLovaszLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SymmetricLovaszLoss, self).__init__()
    def forward(self, logits, targets):
        return ((lovasz_hinge(logits, targets, per_image=True)) \
                + (lovasz_hinge(-logits, 1-targets, per_image=True))) / 2