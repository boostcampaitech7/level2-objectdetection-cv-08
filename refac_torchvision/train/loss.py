"""Pytorch implementation of Class-Balanced-Loss
   Reference: "Class-Balanced Loss Based on Effective Number of Samples" 
   Authors: Yin Cui and
               Menglin Jia and
               Tsung Yi Lin and
               Yang Song and
               Serge J. Belongie
   https://arxiv.org/abs/1901.05555, CVPR'19.
"""

import numpy as np
import torch
import torch.nn.functional as F

# Focal Loss
def focal_loss(labels, logits, alpha, gamma):
    """
    Focal Loss 계산 함수
    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    
    Args:
      labels: 실제 레이블
      logits: 모델의 예측값 (logits)
      alpha: 가중치 (클래스 불균형 조정)
      gamma: 어려운 샘플에 더 큰 가중치를 주는 파라미터

    Returns:
      focal_loss: Focal Loss 값
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="none")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 + torch.exp(-logits)))

    loss = modulator * BCLoss
    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss) / torch.sum(labels)
    
    return focal_loss


# Class-Balanced Loss
def CB_loss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    """
    Class Balanced Loss 계산 함수
    
    Args:
      labels: 실제 레이블
      logits: 모델의 예측값
      samples_per_cls: 각 클래스당 샘플 수
      no_of_classes: 클래스 수
      loss_type: 사용할 손실 타입 ("sigmoid", "focal", "softmax")
      beta: Class-Balanced Loss의 하이퍼파라미터
      gamma: Focal Loss의 하이퍼파라미터

    Returns:
      cb_loss: Class-Balanced Loss 값
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float().unsqueeze(0).repeat(labels_one_hot.size(0), 1) * labels_one_hot
    weights = weights.sum(1).unsqueeze(1).repeat(1, no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input=logits, target=labels_one_hot, weight=weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(input=pred, target=labels_one_hot, weight=weights)
    return cb_loss


# GIoU Loss
def giou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Generalized IoU (GIoU) Loss 계산 함수
    Args:
      pred_boxes: 예측된 바운딩 박스
      target_boxes: 실제 바운딩 박스

    Returns:
      giou_loss: GIoU Loss 값
    """
    # IoU 계산
    inter_min = torch.max(pred_boxes[..., :2], target_boxes[..., :2])
    inter_max = torch.min(pred_boxes[..., 2:], target_boxes[..., 2:])
    inter_wh = (inter_max - inter_min).clamp(min=0)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])

    union_area = pred_area + target_area - inter_area + eps
    iou = inter_area / union_area

    # GIoU 추가 계산
    enclosing_min = torch.min(pred_boxes[..., :2], target_boxes[..., :2])
    enclosing_max = torch.max(pred_boxes[..., 2:], target_boxes[..., 2:])
    enclosing_wh = (enclosing_max - enclosing_min).clamp(min=0)
    enclosing_area = enclosing_wh[..., 0] * enclosing_wh[..., 1]

    giou = iou - (enclosing_area - union_area) / enclosing_area
    return 1.0 - giou


# DIoU Loss
def diou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Distance IoU (DIoU) Loss 계산 함수
    Args:
      pred_boxes: 예측된 바운딩 박스
      target_boxes: 실제 바운딩 박스

    Returns:
      diou_loss: DIoU Loss 값
    """
    # IoU 계산
    iou = giou_loss(pred_boxes, target_boxes, eps)

    # 중심 좌표 계산
    pred_center = (pred_boxes[..., :2] + pred_boxes[..., 2:]) / 2
    target_center = (target_boxes[..., :2] + target_boxes[..., 2:]) / 2

    # 중심 간의 거리 계산
    center_distance = torch.sum((pred_center - target_center) ** 2, dim=-1)

    # 외접 사각형 계산
    enclosing_min = torch.min(pred_boxes[..., :2], target_boxes[..., :2])
    enclosing_max = torch.max(pred_boxes[..., 2:], target_boxes[..., 2:])
    enclosing_wh = (enclosing_max - enclosing_min).clamp(min=0)
    diagonal_length = torch.sum(enclosing_wh ** 2, dim=-1)

    # DIoU 계산
    diou = iou - center_distance / (diagonal_length + eps)
    return diou


# CIoU Loss
def ciou_loss(pred_boxes, target_boxes, eps=1e-7):
    """
    Complete IoU (CIoU) Loss 계산 함수
    Args:
      pred_boxes: 예측된 바운딩 박스
      target_boxes: 실제 바운딩 박스

    Returns:
      ciou_loss: CIoU Loss 값
    """
    iou = diou_loss(pred_boxes, target_boxes, eps)

    # 너비, 높이 비율
    pred_wh = pred_boxes[..., 2:] - pred_boxes[..., :2]
    target_wh = target_boxes[..., 2:] - target_boxes[..., :2]

    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(pred_wh[..., 0] / (pred_wh[..., 1] + eps)) -
                                       torch.atan(target_wh[..., 0] / (target_wh[..., 1] + eps)), 2)
    alpha = v / (1 - iou + v + eps)

    # CIoU 계산
    ciou = iou - alpha * v
    return 1.0 - ciou
