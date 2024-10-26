
# 🏆 재활용 품목 분류를 위한 Object Detection

## 🥇 팀 구성원

#### 박재우, 이상진, 유희석, 정지훈, 천유동, 임용섭


## 프로젝트 소개
바야흐로 대량 생산, 대량 소비의 시대. 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기 대란', '매립지 부족'과 같은 여러 사회 문제를 낳고 있습니다.

분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

따라서 우리는 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다.

여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 

이번 프로젝트는 `부스트캠프 AI Tech 7기` CV 트랙 내에서 진행된 대회이며 `mAP50(Mean Average Precision)`를 통해 최종평가를 진행하였습니다.

<br />

## 📅 프로젝트 일정
프로젝트 전체 일정

- 2024.09.30(목) ~ 2024.10.24(목)

<br />

## 🥈 프로젝트 결과
### Public
- **18** / 24
- mAP50 : **0.6877**
### Private
- **18** / 24
- mAP50 : **0.6754**

<br />

## 🥉 데이터셋 구조
```
 dataset/
 ├── train.json
 ├── test.json
 ├── test
 │   └─ images
 └── train
     └─ images
 
```
이 코드는 `부스트캠프 AI Tech`에서 제공하는 데이터셋으로 다음과 같은 구성을 따릅니다. 
- 전체 이미지 개수 : 9754장
- 이미지 크기 : (1024, 1024)
- 분류 클래스(10개) : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 전체 데이터 중 학습데이터 4883장, 평가데이터 4871장으로 사용
- 학습 데이터 형식: COCO format
- 제출 형식 : Pascal VOC format csv 파일
<br />

## 🥉 프로젝트 구조
```
project/
│   README.md
│   requirements.txt
│   start_ngrok.py
│
├───EDA
│       CV08_EDA.pdf
│
├───mmdetection
│   ├───configs
│   │   ├───a_custom
│   │             train_base_cascade.py
│   │             train_base_cascade_swin_L.py
│   │             train_base_ddq.py
│   │             train_base_deformable_detr.py
│   │             train_base_dino.py
│   │             train_base_dino_val.py
│   │             train_base_efficientNet.py
│   │             train_base_faster_rcnn.py
│   │             train_base_retinanet_swin_L.py
│   ├───tools
│            fold_train.py
│
├───refac_torchvision
│   │   main.py
│   │
│   ├───data
│   │       custom_dataset.py
│   │       transforms.py
│   │
│   ├───models
│   │       model.py
│   │       save_load.py
│   │
│   ├───process
│   │       img_process.py
│   │
│   └───train
│           eval.py
│           loss.py
│           train.py
│           utils.py
│
├───tools
│       cleansing_labels.ipynb
│       coco2yolo.py
│       csv_bbox_visualize.ipynb
│       ensemble.ipynb
│       json_bbox_visualize.ipynb
│       json_coco2pascal.ipynb
│
└───yolo
    │   yolo_inference.ipynb
    │   yolo_train.ipynb
    │
    └───cfg
            coco-trash.yaml
```
### 1) MMDetection
- `configs/a_custom/`: MMDetection 모델의 학습과 추론에 필요한 설정 파일들을 포함하고 있습니다.
- `tools/fold_train.py`: Stratified Group K-Fold 교차 검증을 통한 학습을 위한 스크립트입니다.

### 2) refac_torchvision
- `data/`: 데이터셋 로딩 및 전처리 관련 코드를 포함합니다.
- `models/`: 모델 로드 및 저장 관련 파일을 포함합니다.
- `process/`: 이미지 전처리 기능을 제공합니다.
- `train/`: 모델 학습 및 평가에 필요한 기능들을 제공합니다.

### 3) tools
- `cleansing_labels.ipynb`: 레이블 클렌징 작업을 수행하는 노트북입니다.
- `coco2yolo.py`: COCO 형식의 데이터셋을 YOLO 형식으로 변환하는 스크립트입니다.
- `csv_bbox_visualize.ipynb`, `json_bbox_visualize.ipynb`: 바운딩 박스 시각화를 위한 노트북 파일입니다.
- `json_coco2pascal.ipynb`: COCO 형식의 JSON 파일을 Pascal VOC 형식으로 변환하고 CSV로 저장합니다.
- `ensemble.ipynb`: CSV 형식으로 출력된 추론 결과 파일들을 앙상블(NMS, WBF) 및 NMW(Non-Maximum Weighted) 방식을 적용하여 최적화합니다.

### 4) YOLO
- `yolo_inference.ipynb`: YOLO 모델을 이용해 추론 작업을 수행하는 노트북 파일입니다.
- `yolo_train.ipynb`: YOLO 모델의 학습을 위한 노트북 파일입니다.
- `cfg/`: YOLO 학습 설정을 위한 구성 파일들을 포함합니다.

<br />

## ⚙️ 설치

### Dependencies
이 모델은 Tesla v100 32GB의 환경에서 작성 및 테스트 되었습니다.
또 모델 실행에는 다음과 같은 외부 라이브러리가 필요합니다.

```bash
pip install -r requirements.txt
mim install mmcv-full==1.7.0
mim install mmcv==2.1.0
mim install mmengine
```

- visdom==0.2.4
- seaborn==0.12.2
- albumentations==0.4.6
- imgaug==0.4.0
- pycocotools==2.0.6
- opencv-python==4.7.0.72
- tqdm==4.65.0
- torchnet==0.0.4
- pandas
- map-boxes==1.0.5
- jupyter==1.0.0
- openmim
- mmdet==3.3.0

<br />

## 🚀 빠른 시작
### Train
#### MMDetection

```python
# fold train
python tools/fold_train.py {config_path}

# train
python tools/train.py {config_path}
```

#### Torchvision
```python
python main.py
```
##### Torchvision Parser
기본 설정
- `--annotations_path` : train.json path
- `--data_dir` : Dataset directory
- `--model_name` : 학습 진행할 모델 이름 ( 기본값: Faster RCNN )
- `--device` : `cuda` or `cup` ( 기본값 : cuda )
- `--base_dir` : result path

학습 설정
- `--num_epochs` : 학습할 에폭 수 (기본값 : 1)
- `--batch_size` : 배치 크기 결정 ( 기본값 : 32 )
- `--n_split` : fold split 수량 ( 기본값 : 5 )
- `--training_mode` : `standard` or `fold` (필수)

옵티마이저 설정
- `--optimizer` : `SGD` or `AdamW` ( 기본값 : SGD )
- `--learning_rate` : 학습률 설정 ( 기본값 : 0.001)
- `--momentum` : SGD Momentum 값 설정 ( 기본값 0.9 )
- `--weight_decay` : 옵티마이저 weight decay 설정 ( 기본값 : 0.0009 )

스케쥴러 설정 (CosineAnnealing)
- `--scheduler_t_max` : 코사인 어널링 t max 설정 ( 기본값 : 40)
- `--scheduler_eta_min` : 코사인 어널링 eta min 설정 ( 기본값 : 0)

### Test
#### MMDetection
```python
python tools/test.py {config_path} {pth_file_path}
```

##### MMDetection Parser
- `--tta` : Test Time Augmentation 활성화
<br />

## 🏅 Wrap-Up Report   
### [ Wrap-Up Report 👑]
