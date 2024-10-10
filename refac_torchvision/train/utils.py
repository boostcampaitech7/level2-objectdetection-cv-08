import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import label_binarize
import matplotlib.patches as patches
from PIL import Image
import io

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

    # 여백을 최소화한 상태로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    plt.close(fig)
    return Image.open(buf)

# 시드 설정 함수: 실험 재현성을 보장하기 위해 시드 설정
def set_seed(seed=42):
    random.seed(seed)  # Python 랜덤 시드
    np.random.seed(seed)  # NumPy 랜덤 시드
    torch.manual_seed(seed)  # PyTorch에서 CPU 시드 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch에서 CUDA 시드 고정
        torch.cuda.manual_seed_all(seed)  # 다중 GPU 사용 시 모든 GPU에 시드 고정

# StratifiedGroupKFold로 데이터를 나누는 함수
def stratified_group_kfold_split(dataset, n_splits=5, random_state=42):
    """
    dataset: CustomDataset 객체 (이미지 ID와 레이블을 포함)
    """
    image_ids = [img['id'] for img in dataset.coco.dataset['images']]
    labels = []
    groups = np.array(image_ids)  # 각 이미지가 그룹으로 간주되며, numpy 배열로 변환

    # 각 이미지에 대한 레이블을 추출 (하나의 이미지에 여러 주석이 있을 수 있음)
    for img_id in image_ids:
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id)
        anns = dataset.coco.loadAnns(ann_ids)
        if anns:
            labels.append(anns[0]['category_id'])  # 이미지의 첫 번째 주석의 category_id를 레이블로 사용

    # X는 각 이미지 ID를 가진 1차원 배열로 설정
    X = np.ones((len(image_ids), 1))
    y = np.array(labels)  # 이미지 레벨의 레이블

    sgkf = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for train_idx, val_idx in sgkf.split(X, y, groups):
        yield train_idx, val_idx

# 결과 폴더 생성 함수
def create_result_folder(base_dir, model_name, num_epochs):
    result_dir = os.path.join(base_dir, f"{model_name}_{num_epochs}epoch")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir

# mAP 곡선 저장 함수
def save_map_curve(map_values, result_dir):
    plt.figure()
    plt.plot(map_values)
    plt.xlabel('Epochs')
    plt.ylabel('mAP')
    plt.title('mAP Curve')
    plt.savefig(os.path.join(result_dir, 'mAP_curve.png'))
    plt.close()