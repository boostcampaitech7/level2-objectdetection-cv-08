import torch
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import cv2
import os
from process.img_process import process_img, filter_small_bboxes

class CustomDataset(Dataset):
    def __init__(self, annotation, data_dir, transforms=None, apply_processing=False, filter_bbox=False):
        """
        CustomDataset 클래스 초기화
        Args:
            annotation (str): COCO 형식의 주석 파일 경로
            data_dir (str): 이미지 디렉토리 경로
            transforms (callable, optional): 이미지와 바운딩박스 변환 함수
            apply_processing (bool, optional): 이미지 전처리 여부
            filter_bbox (bool, optional): 작은 바운딩 박스 필터링 여부
        """
        super().__init__()
        self.data_dir = data_dir
        self.coco = COCO(annotation)
        self.transforms = transforms
        self.apply_processing = apply_processing
        self.filter_bbox = filter_bbox
        self.image_ids = self.coco.getImgIds()  # 이미지 ID 리스트를 미리 가져옵니다.

    def __getitem__(self, index: int):
        """
        데이터셋에서 index에 해당하는 이미지와 타겟(바운딩 박스 정보)를 반환합니다.
        Args:
            index (int): 데이터셋의 인덱스
        Returns:
            Tuple[torch.Tensor, dict]: 이미지와 타겟 정보
        """
        image_id = self.image_ids[index]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.data_dir, image_info['file_name'])

        image = self.load_image(image_path)
        target = self.load_target(image_id)

        if self.filter_bbox:
            target['boxes'], target['labels'] = self.apply_bbox_filter(image, target['boxes'], target['labels'])

        if self.apply_processing:
            image = process_img(image)

        if self.transforms:
            image, target = self.apply_transforms(image, target)

        return image, target

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image / 255.0

    def load_target(self, image_id):
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        boxes = np.array([x['bbox'] for x in anns], dtype=np.float32)
        boxes[:, 2:] += boxes[:, :2]
        
        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor([x['category_id'] + 1 for x in anns], dtype=torch.int64),
            'image_id': torch.tensor([image_id]),
            'area': torch.as_tensor([x['area'] for x in anns], dtype=torch.float32),
            'iscrowd': torch.as_tensor([x['iscrowd'] for x in anns], dtype=torch.int64)
        }
        return target

    def apply_bbox_filter(self, image, boxes, labels):
        """
        작은 바운딩 박스를 필터링합니다.
        Args:
            image (np.ndarray): 이미지
            boxes (np.ndarray): 바운딩 박스 리스트
            labels (np.ndarray): 각 바운딩 박스의 레이블 리스트
        Returns:
            Tuple[np.ndarray, np.ndarray]: 필터링된 바운딩 박스와 레이블 리스트
        """
        return filter_small_bboxes(image, boxes.numpy(), labels.numpy(), min_area_ratio=0.007)

    def apply_transforms(self, image, target):
        """
        이미지와 바운딩 박스에 변환을 적용합니다.
        Args:
            image (np.ndarray): 이미지
            target (dict): 타겟 정보 (바운딩 박스와 레이블)
        Returns:
            Tuple[np.ndarray, dict]: 변환된 이미지와 타겟 정보
        """
        sample = {'image': image, 'bboxes': target['boxes'], 'labels': target['labels']}
        sample = self.transforms(**sample)
        image = sample['image']
        target['boxes'] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
        return image, target

    def __len__(self):
        return len(self.image_ids)
