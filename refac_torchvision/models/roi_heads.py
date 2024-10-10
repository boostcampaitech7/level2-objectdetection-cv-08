import torch.nn.functional as F
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch

class CustomFastRCNNPredictor(FastRCNNPredictor):
    def __init__(self, in_channels, num_classes):
        super().__init__(in_channels, num_classes)
    
    def forward(self, x):
        print("Forward pass in CustomFastRCNNPredictor")
        class_logits, box_regression = super().forward(x)
        return class_logits, box_regression

    def compute_loss(self, class_logits, box_regression, labels, regression_targets):
        """
        labels: list of Tensors, each with shape (num_boxes,)
        regression_targets: list of Tensors, each with shape (num_boxes, 4)
        """
        # Concatenate all labels and regression targets into a single Tensor
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        #class별 가중치 주기//맨처음 배경
        class_num  = [0, 3996, 6352, 897, 936, 982, 2943, 1263, 5178, 159, 468]
        total_samples = sum(class_num)
        weights = [total_samples / (num + 1e-6) for num in class_num]
        weight_tensor = torch.tensor(weights, device=class_logits.device)

        # 분류 손실 (Cross Entropy)
        print("Custom Loss!!!!")
        # classification_loss = F.cross_entropy(class_logits, labels, weight=weight_tensor)
        classification_loss = F.cross_entropy(class_logits, labels)

        sampled_pos_inds_subset = torch.where(labels > 0)[0]
        labels_pos = labels[sampled_pos_inds_subset]
        
        N, num_classes = class_logits.shape
        box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

        # 회귀 손실 (Smooth L1 Loss)
        box_loss = F.smooth_l1_loss(
            box_regression[sampled_pos_inds_subset, labels_pos],
            regression_targets[sampled_pos_inds_subset],
            beta=1 / 9,
            reduction="sum"
        )

        box_loss = box_loss / labels.numel()  # Normalize by the number of labels

        return classification_loss, box_loss
