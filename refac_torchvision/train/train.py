from train.utils import Averager, stratified_group_kfold_split, collate_fn
from models.save_load import save_model
from models.model import create_model
import torch
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
import os

# 학습 스텝 함수
def train_epoch(train_data_loader, optimizer, model, device, scaler):
    model.train()
    train_loss_hist = Averager()
    train_loss_hist.reset()

    for images, targets in tqdm(train_data_loader, desc="Training"):
        images = list(image.float().to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        valid_images, valid_targets = [], []
        for img, tgt in zip(images, targets):
            if tgt['boxes'].shape[0] > 0:
                valid_images.append(img)
                valid_targets.append(tgt)

        if len(valid_targets) == 0:
            continue

        with autocast():
            loss_dict = model(valid_images, valid_targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

        train_loss_hist.send(loss_value)
        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

    return train_loss_hist.value

# 검증 스텝 함수
def validate_step(val_data_loader, model, coco, device):
    model.eval()
    coco_results = []

    with torch.no_grad():
        for images, targets in tqdm(val_data_loader, desc="Validating"):
            images = [image.to(device) for image in images]
            outputs = model(images)

            for i, output in enumerate(outputs):
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    coco_results.append({
                        'image_id': int(targets[i]['image_id']),
                        'category_id': int(label) - 1,
                        'bbox': box.tolist(),
                        'score': float(score)
                    })

    if coco:
        coco_dt = coco.loadRes(coco_results)
        coco_eval = COCOeval(coco, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        coco_metrics = {
            "mAP@0.5:0.95": coco_eval.stats[0],
            "mAP@0.5": coco_eval.stats[1],
            "mAP@0.75": coco_eval.stats[2],
            "mAP_small": coco_eval.stats[3],
            "mAP_medium": coco_eval.stats[4],
            "mAP_large": coco_eval.stats[5],
            "AR@1": coco_eval.stats[6],
            "AR@10": coco_eval.stats[7],
            "AR@100": coco_eval.stats[8],
            "AR@100_small": coco_eval.stats[9],
            "AR@100_medium": coco_eval.stats[10],
            "AR@100_large": coco_eval.stats[11],
        }
    else:
        coco_metrics = {"mAP@0.5:0.95": 0.0}

    return coco_metrics

# 일반 학습 실행 함수
def run_standard_training(train_data_loader, val_data_loader, model, optimizer, scheduler, scaler, device, base_dir, num_epochs):
    best_map = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(train_data_loader, optimizer, model, device, scaler)
        validation_metrics = validate_step(val_data_loader, model, None, device)
        scheduler.step()

        print(f"Epoch #{epoch + 1} | Train Loss: {train_loss:.4f}")
        print("Validation Metrics:")
        for metric, value in validation_metrics.items():
            print(f"  {metric}: {value:.4f}")

        if validation_metrics["mAP@0.5"] > best_map:
            best_map = validation_metrics["mAP@0.5"]
            model_save_path = os.path.join(base_dir, "best_model.pth")
            save_model(model, model_save_path)
            print(f"Best model saved to {model_save_path}")

# K-Fold 학습 실행 함수
def run_fold_training(train_dataset, model_name, device, base_dir, num_epochs, n_splits, batch_size, learning_rate, momentum, weight_decay):
    best_global_map = 0
    best_global_model_path = ""

    for fold_idx, (train_idx, val_idx) in enumerate(stratified_group_kfold_split(train_dataset, n_splits=n_splits)):
        print(f"Starting fold {fold_idx + 1}/{n_splits}")
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)

        train_data_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_data_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

        model = create_model(model_name, num_classes=11).to(device)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0)
        scaler = GradScaler()
        
        fold_best_map = 0
        for epoch in range(num_epochs):
            train_loss = train_epoch(train_data_loader, optimizer, model, device, scaler)
            validation_metrics = validate_step(val_data_loader, model, None, device)
            scheduler.step()

            print(f"Fold {fold_idx + 1}/{n_splits}, Epoch #{epoch + 1} | Train Loss: {train_loss:.4f}")
            print("Validation Metrics:")
            for metric, value in validation_metrics.items():
                print(f"  {metric}: {value:.4f}")

            if validation_metrics["mAP@0.5"] > fold_best_map:
                fold_best_map = validation_metrics["mAP@0.5"]
                fold_best_model_path = os.path.join(base_dir, f"{model_name}_fold{fold_idx}_best.pth")
                save_model(model, fold_best_model_path)
                print(f"Best model saved to {fold_best_model_path}")

        if fold_best_map > best_global_map:
            best_global_map = fold_best_map
            best_global_model_path = fold_best_model_path

    print(f"Best model path across folds: {best_global_model_path} with mAP@0.5: {best_global_map:.4f}")