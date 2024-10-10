import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import os
from process.img_process import process_img,filter_small_bboxes

class CustomDataset(Dataset):
    def __init__(self, annotation, data_dir, transforms=None, apply_processing=False, filter_bbox=False):
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.transforms = transforms
        self.apply_processing = apply_processing
        self.filter_bbox = filter_bbox

        self.image_ids = self.coco.getImgIds()  # 이미지 ID 리스트를 미리 가져옵니다.

    def __getitem__(self, index: int):
        # 인덱스를 통해 실제 COCO 이미지 ID를 가져옵니다.
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]  # 이미지 ID로 COCO 정보 로드
        image_path = os.path.join(self.data_dir, image_info['file_name'])
        image = self.load_image(image_path)
        target = self.load_target(image_id)

        # 작은 바운딩박스 제거 하단 주석 제거
        if self.filter_bbox:
            bboxes = target['boxes'].numpy()
            labels = target['labels'].numpy()
            filtered_bboxes, filtered_labels = filter_small_bboxes(image, bboxes, labels, min_area_ratio=0.007)
            target['boxes'] = torch.as_tensor(filtered_bboxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(filtered_labels, dtype=torch.int64)

        # dbscan을 이용한 겹치는 bbox 제거 하단 주석 제거
        # min_sample : 클러스터를 형성하기 위한 최소한의 포인트
        # esp : 각 샘플에 대해 주변 데이터 포인트 중에서 같은 클러스터에 포함될 수 있는 최대 거리

        if self.apply_processing:
            image = process_img(image)

        if self.transforms:
            image, target = self.apply_transforms(image, target)

        return image,target

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image / 255.0

    def load_target(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = np.array([x['bbox'] for x in anns], dtype=np.float32)
        boxes[:, 2:] += boxes[:, :2]  # (x_min, y_min, w, h) -> (x_min, y_min, x_max, y_max)
        
        labels = torch.as_tensor([x['category_id'] + 1 for x in anns], dtype=torch.int64)
        areas = torch.as_tensor([x['area'] for x in anns], dtype=torch.float32)
        iscrowds = torch.as_tensor([x['iscrowd'] for x in anns], dtype=torch.int64)

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowds
        }
        return target

    def apply_transforms(self, image, target):
        sample = {'image': image, 'bboxes': target['boxes'], 'labels': target['labels']}
        sample = self.transforms(**sample)
        image = sample['image']
        target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
        return image, target

    def __len__(self):
        return len(self.image_ids)  # 이미지 ID 리스트의 길이 반환
