from train.utils import set_seed, stratified_group_kfold_split,save_map_curve
from data.custom_dataset import CustomDataset
from data.transforms import get_transform
from models.save_load import save_model
from train.train import train_fn
from torch.utils.data import DataLoader
from train.utils import collate_fn, visualize_image_with_boxes
from models.model import create_model
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import sys

def visualize_box_comparison(annotation, data_dir, index=2, save_path='/data/ephemeral/home/faster_rcnn/refac_torchvision/processing_img_with_boxes.png'):
    train_no_process_dataset = CustomDataset(annotation, data_dir, get_transform(train=False),filter_bbox=False)
    process_train_dataset = CustomDataset(annotation, data_dir, get_transform(train=True),filter_bbox=False)

    image_no_process, target_no_process = train_no_process_dataset.__getitem__(index)
    image_process, target_process = process_train_dataset.__getitem__(index)

    image_no_process = image_no_process.permute(1, 2, 0).numpy()
    image_process = image_process.permute(1, 2, 0).numpy()

    boxes_no_process = target_no_process['boxes'].numpy()
    boxes_process = target_process['boxes'].numpy()

    image_no_process_with_boxes = visualize_image_with_boxes(image_no_process, boxes_no_process)

    image_process_with_boxes = visualize_image_with_boxes(image_process, boxes_process)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_no_process_with_boxes)
    axes[0].set_title('Original')
    axes[1].imshow(image_process_with_boxes)
    axes[1].set_title('Processed')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    sys.exit()

def visualized_process_img(annotation, data_dir, index=2, save_path='/data/ephemeral/home/faster_rcnn/refac_torchvision/processing_img.png'):
    train_no_process_dataset = CustomDataset(annotation, data_dir, get_transform(train=False))
    process_train_dataset = CustomDataset(annotation, data_dir, get_transform(train=True))
    image_no_process, _ = train_no_process_dataset.__getitem__(index)
    image_process, _ = process_train_dataset.__getitem__(index)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1x2 레이아웃, 10x5 인치 크기

    image_no_process = image_no_process.permute(1, 2, 0)
    image_process = image_process.permute(1, 2, 0)

    axes[0].imshow(image_no_process)
    axes[0].set_title('origin')

    axes[1].imshow(image_process)
    axes[1].set_title('processing')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    sys.exit()

def main():
    # 1. 시드 설정
    set_seed(42)

    # 2. 데이터셋 설정
    annotation = '/data/ephemeral/home/dataset/train.json'
    data_dir = '/data/ephemeral/home/dataset'
    
    train_dataset = CustomDataset(annotation, data_dir, get_transform(train=True),filter_bbox=True)
    val_dataset = CustomDataset(annotation, data_dir, get_transform(train=False))

    # visualized_process_img(annotation, data_dir, 2)
    # visualize_box_comparison(annotation, data_dir, 5) # 너무 작은 bbox 제거

    # 3. 장치 설정
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    # 4. 모델 생성 및 최종 best 모델 추적을 위한 변수 설정
    model_name = 'fasterrcnn_resnet50_fpn'
    best_global_map = 0
    best_global_model_path = ""
    
    # 모델 및 Optimizer
    model = create_model(model_name, num_classes=11)
    model.to(device)
    print(model.roi_heads.box_predictor)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=0)

    # 5. K-Fold Cross Validation 설정   
    num_epochs = 1
    n_splits = 5
    epoch_map = []
    base_dir = f'/data/ephemeral/home/faster_rcnn/refac_torchvision/results/{model_name}_epoch{num_epochs}'

    # 에폭마다 모든 폴드를 실행
    for fold_idx, (train_idx, val_idx) in enumerate(stratified_group_kfold_split(train_dataset, n_splits=n_splits)):
        print(f"Starting fold {fold_idx + 1}/{n_splits}")

        # DataLoader 설정
        train_subset = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(val_dataset, val_idx)

        train_data_loader = DataLoader(train_subset, batch_size=16, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_data_loader = DataLoader(val_subset, batch_size=16, shuffle=False, num_workers=4, collate_fn=collate_fn)

        fold_best_map = 0  # 각 폴드에서 최고 mAP 기록
        fold_epoch_map = []  # 폴드 내 에폭별 mAP 기록

        # 각 폴드별로 모든 에폭을 실행
        for epoch in range(num_epochs):
            print(f"Fold {fold_idx + 1}/{n_splits}, Epoch {epoch + 1}/{num_epochs}")

            # 각 에폭에 대해 학습 및 검증
            current_map = train_fn(model_name,
                                   train_data_loader,
                                    val_data_loader, 
                                    optimizer, 
                                    model, 
                                    val_dataset.coco, 
                                    device, 
                                    base_dir, 
                                    fold_idx, 
                                    epoch)

            # 폴드 내 에폭별 mAP 기록
            fold_epoch_map.append(current_map)

            scheduler.step()

            # 폴드 내에서 최고 mAP 모델 저장
            fold_best_model_path = os.path.join(base_dir, f"{model_name}_fold{fold_idx}", f"checkpoint_epoch{epoch + 1}.pth")
            if current_map > fold_best_map:
                fold_best_map = current_map
                save_model(model, fold_best_model_path)
        
        # 폴드별 평균 mAP 기록
        average_fold_map = np.mean(fold_epoch_map)
        epoch_map.append(average_fold_map)
        print(f"Fold {fold_idx + 1} Average mAP: {average_fold_map:.4f}")

        # 폴드별 최적 mAP 모델 경로 기록
        if fold_best_map > best_global_map:
            best_global_map = fold_best_map
            best_global_model_path = fold_best_model_path

    # mAP 곡선 저장
    save_map_curve(epoch_map, base_dir)

    print(best_global_model_path)
    
    # 최종 best 모델 저장
    final_model_path = os.path.join(base_dir, "best_model.pth")
    final_model_dir = os.path.dirname(final_model_path)

    # 디렉토리가 없으면 생성
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)

    # best_global_model_path가 존재하는지 확인 후 이동
    if os.path.exists(best_global_model_path):
        print(f"Best model found at {best_global_model_path} with mAP {best_global_map:.4f}")
        os.rename(best_global_model_path, final_model_path)
    else:
        print(f"Best model not found at {best_global_model_path}")

if __name__ == '__main__':
    main()
