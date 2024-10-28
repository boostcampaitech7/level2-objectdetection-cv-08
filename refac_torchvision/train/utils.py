import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.patches as patches
from PIL import Image
import io
import sys
from data.custom_dataset import CustomDataset
from data.transforms import get_transform

def collate_fn(batch):
    return tuple(zip(*batch))

# Averager 클래스: 손실 값을 평균내는 클래스
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        return self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0

# 바운딩 박스랑 이미지 플롯
def visualize_image_with_boxes(image, boxes):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box in boxes:
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    plt.close(fig)
    return Image.open(buf)


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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def stratified_group_kfold_split(dataset, n_splits=5, random_state=42):
    image_ids = [img['id'] for img in dataset.coco.dataset['images']]
    labels = []
    groups = np.array(image_ids)

    for img_id in image_ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        if anns:
            labels.append(anns[0]['category_id'])

    X = np.ones((len(image_ids), 1))
    y = np.array(labels)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_idx, val_idx in sgkf.split(X, y, groups):
        yield train_idx, val_idx

def create_result_folder(base_dir, model_name, num_epochs):
    result_dir = os.path.join(base_dir, f"{model_name}_{num_epochs}epoch")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir