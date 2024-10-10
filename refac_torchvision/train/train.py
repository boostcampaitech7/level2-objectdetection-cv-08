from train.utils import (
    create_result_folder, save_map_curve, Averager)
from models.save_load import save_model
from train.eval import evaluate_fn
import torch
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from train.metric import compute_map_over_iou_ranges
from torch.cuda.amp import autocast, GradScaler

# 학습 스텝 함수
def train_epoch(train_data_loader, optimizer, model, device,scaler):
    model.train()
    train_loss_hist = Averager()  # 손실을 기록할 Averager 클래스
    train_loss_hist.reset()

    # Training loop
    for images, targets in tqdm(train_data_loader, desc="Training"):
        images = list(image.float().to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # bbox 제거 시 사용
        valid_images = []
        valid_targets = []
        for img, tgt in zip(images, targets):
            if tgt['boxes'].shape[0] > 0:
                valid_images.append(img)
                valid_targets.append(tgt)

        if len(valid_targets) == 0:
            continue

        # 모델의 예측 및 손실 계산
        # loss_dict = model(images, targets) 원본
        # mixed precision
        with autocast():
            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

        train_loss_hist.send(loss_value)

        # Optimizer 스텝
        optimizer.zero_grad()

        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

    return train_loss_hist.value  # 에포크 동안의 평균 손실 반환

def validate_step(val_data_loader, model, coco, device):
    model.eval()
    # coco_results = []  # mAP를 위한 예측 결과 저장
    ground_truths = []
    detections = []

    with torch.no_grad():
        for images, targets in tqdm(val_data_loader, desc="Validating"):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 검증 시에는 손실을 계산하지 않고, 예측 결과만 수집
            outputs = model(images)
            for i, output in enumerate(outputs):
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()

                # 실제 값 수집
                for gt_box in gt_boxes:
                    ground_truths.append({
                        'image_id': targets[i]['image_id'],
                        'bbox': gt_box
                    })

                # 예측 값 수집
                for box, score, label in zip(pred_boxes, scores, labels):
                    detections.append({
                        'image_id': targets[i]['image_id'],
                        'bbox': box,
                        'score': score,
                        'label': label
                    })

    # IoU 임계값별 mAP 계산
    mean_ap, precision_list, recall_list = compute_map_over_iou_ranges(ground_truths, detections)
    
    print(f"mAP: {mean_ap:.4f}")
    print(f"Precision at IoU thresholds: {precision_list}")
    print(f"Recall at IoU thresholds: {recall_list}")

    #         for i, output in enumerate(outputs):
    #             boxes = output['boxes'].cpu().numpy()
    #             boxes[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    #             boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
    #             scores = output['scores'].cpu().numpy()
    #             labels = output['labels'].cpu().numpy()
    #             for box, score, label in zip(boxes, scores, labels):
    #                 coco_results.append({
    #                     'image_id': int(targets[i]['image_id']),
    #                     'category_id': int(label)-1,
    #                     'bbox': box.tolist(),
    #                     'score': float(score)
    #                 })

    # # COCO API로 mAP 계산
    # coco_dt = coco.loadRes(coco_results)
    # coco_eval = COCOeval(coco, coco_dt, 'bbox')
    # coco_eval.evaluate()
    # coco_eval.accumulate()
    # coco_eval.summarize()

    # current_map = coco_eval.stats[0]  # mAP 0.5:0.95

    return mean_ap  # 검증 손실 대신 mAP만 반환

# 학습 및 검증을 진행하는 최종 함수
def train_fn(model_name, train_data_loader, val_data_loader, optimizer, model, coco, device, base_dir, fold_idx,epoch):
    train_losses = [] # 손실과 mAP 기록 리스트
    scaler = GradScaler()

    # 1. 학습 스텝
    train_loss = train_epoch(train_data_loader, optimizer, model, device,scaler)
    train_losses.append(train_loss)

    # 2. 검증 스텝 (mAP 계산)
    current_map = validate_step(val_data_loader, model, coco, device)
    print(f"Epoch #{epoch + 1} | Train Loss: {train_loss:.4f}, mAP: {current_map:.4f}")
    return current_map
