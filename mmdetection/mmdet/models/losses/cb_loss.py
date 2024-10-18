"""
Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.registry import MODELS

@MODELS.register_module()
class ClassBalancedLoss(nn.Module):
    """logits와 실제 레이블 `labels` 사이의 Class Balanced Loss를 계산하는 함수.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    여기서 Loss는 신경망에서 사용되는 표준 손실 중 하나를 의미.

    인자:
      labels: [배치] 크기의 int 텐서. 실제 레이블.
      logits: [배치, 클래스 수] 크기의 float 텐서. 모델의 예측값.
      samples_per_cls: [클래스 수] 크기의 각 클래스당 샘플 수를 나타내는 파이썬 리스트.
      no_of_classes: 총 클래스 수를 나타내는 int 값.
      loss_type: string. "sigmoid", "focal", "softmax" 중 하나를 선택.
      beta: Class Balanced Loss에 사용되는 하이퍼파라미터 (float).
      gamma: Focal Loss에 사용되는 하이퍼파라미터 (float).

    반환값:
      cb_loss: class balanced loss를 나타내는 float 텐서.
    """
    def __init__(self, samples_per_cls, no_of_classes, beta=0.9999, gamma=2.0, loss_type='focal'):
        super(ClassBalancedLoss, self).__init__()
        self.samples_per_cls = samples_per_cls
        self.no_of_classes = no_of_classes
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

    def forward(self, labels, logits):
        effective_num = 1.0 - np.power(self.beta, self.samples_per_cls)
        weights = (1.0 - self.beta) / np.sum(effective_num)
        weights = weights / np.sum(weights) * self.no_of_classes

        labels_one_hot = F.one_hot(labels, self.no_of_classes).float()

        weights = torch.tensor(weights).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.no_of_classes)

        if self.loss_type == 'focal':
            cb_loss = self.focal_loss(labels_one_hot, logits, weights, self.gamma)
        elif self.loss_type == 'sigmoid':
            cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weights=weights)
        elif self.loss_type == 'softmax':
            pred = logits.softmax(dim=1)
            cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
        return cb_loss

    def focal_loss(self, labels, logits, alpha, gamma):
        BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction='none')

        if gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-1.0 * logits)))

        loss = modulator * BCLoss
        weighted_loss = alpha * loss
        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(labels)
        return focal_loss