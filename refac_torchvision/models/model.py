import torch
import torchvision.models.detection as detection_models
from models.roi_heads import CustomFastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(model_name, num_classes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_name == 'fasterrcnn_resnet50_fpn':
        model = detection_models.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = CustomFastRCNNPredictor(in_features, num_classes)
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    elif model_name == 'retinanet_resnet50_fpn_v2':
        model = detection_models.retinanet_resnet50_fpn_v2(pretrined=True)
        model.head = RetinaNetHead(in_channels=model.backbone.out_channels,
                                   num_anchors=model.head.classification_head.num_anchors,
                                   num_classes=num_classes)

    elif model_name == 'maskrcnn_resnet50_fpn':
        model = detection_models.maskrcnn_resnet50_fpn(pretrined=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = detection_models.FastRCNNPredictor(in_features, num_classes)
        
    elif model_name == 'keypointrcnn_resnet50_fpn':
        model = detection_models.keypointrcnn_resnet50_fpn(pretrined=True)
        in_features = model.roi_heads.keypoint_predictor.kps_score_lowres.in_features
        model.roi_heads.keypoint_predictor = detection_models.KeypointRCNNPredictor(in_features, num_classes)
    
    else:
        raise ValueError(f"지원하지 않는 모델: {model_name}")

    return model.to(device)
