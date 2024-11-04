import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transform(train=True):

    bbox_params = {'format': 'pascal_voc', 'label_fields': ['labels']}

    if train:
        return A.Compose([
            A.Resize(2048,2048,interpolation=cv2.INTER_LANCZOS4),
            A.Flip(p=0.5),
            ToTensorV2()
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2()
        ], bbox_params=bbox_params)