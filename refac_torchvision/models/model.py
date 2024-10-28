import torch
import torchvision.models.detection as detection_models
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

SUPPORTED_MODELS = ['fasterrcnn_resnet50_fpn', 'retinanet_resnet50_fpn_v2', 'maskrcnn_resnet50_fpn', 'keypointrcnn_resnet50_fpn']

def create_model(model_name, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    if model_name == 'fasterrcnn_resnet50_fpn':
        model = initialize_fasterrcnn(num_classes)

    elif model_name == 'retinanet_resnet50_fpn_v2':
        model = initialize_retinanet(num_classes)

    elif model_name == 'maskrcnn_resnet50_fpn':
        model = initialize_maskrcnn(num_classes)
        
    elif model_name == 'keypointrcnn_resnet50_fpn':
        model = initialize_keypointrcnn(num_classes)

    return model.to(device)

def initialize_fasterrcnn(num_classes):
    model = detection_models.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def initialize_retinanet(num_classes):
    model = detection_models.retinanet_resnet50_fpn_v2(pretrained=True)
    model.head = RetinaNetHead(
        in_channels=model.backbone.out_channels,
        num_anchors=model.head.classification_head.num_anchors,
        num_classes=num_classes
    )
    return model

def initialize_maskrcnn(num_classes):
    model = detection_models.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def initialize_keypointrcnn(num_classes):
    model = detection_models.keypointrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_features
    model.roi_heads.keypoint_predictor = detection_models.KeypointRCNNPredictor(in_features, num_classes)
    return model
