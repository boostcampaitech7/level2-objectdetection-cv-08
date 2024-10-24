
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
-  / 24
- mAP50 : 
### Private
-  / 24
- mAP50 : 

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
- 제출 형식 : Pascal VOC format, .csv 파일
<br />

# `여기까지 수정`

## 🥉 프로젝트 구조
```
project/
.
|-- mmdetection
|-- EDA
|   `-- CV08_EDA.pdf
|-- README.md
|-- refac_torchvision
|   |-- data
|   |-- main.py
|   |-- models
|   |-- process
|   `-- train
|-- requirements.txt
|-- start_ngrok.py
|-- tools
|   |-- bbox_visualize.ipynb
|   |-- coco2yolo.py
|   |-- ensemble.ipynb
|   |-- json_bbox_check.ipynb
|   |-- jsontocsv.ipynb
|   |-- offline_score_filter.ipynb
|   |-- train_bbox_count.ipynb
|   `-- unique_ids.ipynb
`-- yolo
    |-- cfg
    |-- yolo_inference.ipynb
    `-- yolo_train.ipynb
```

### 1) `configs`
- 설정 파일을 관리하는 폴더
- `config_manager.py`는 `config.yaml`을 불러와 학습 및 추론에 필요한 설정을 관리합니다.
### 2) `datas`
- 데이터 로딩 및 전처리를 담당하는 폴더
- `custom_dataset.py`에서는 커스텀 데이터셋을 정의하며, `cutmix.py`와 같은 데이터 증강 기법도 포함되어 있습니다.
### 3) `models`
- 모델 선택 및 초기화 로직을 정의하는 폴더
- `model_selector.py`에서 `timm 라이브러리`를 통해 다양한 사전 학습된 모델을 선택하고 사용할 수 있습니다.
### 4) `optimizers`
- 학습 중 Learning Rate 조절을 위한 스케줄러를 정의하는 폴더
- `optimizer.py`에서 `Adam`, `SGD`, `AdamW` 옵티마이저를 설정할 수 있습니다.
### 5) `schedulers`
- 학습 중 Learning Rate 조절을 위한 스케줄러를 정의하는 폴더
- `scheduler.py`는 다양한 학습률 조절 방법을 제공합니다.
### 6) `trainers`
- 학습과 추론에 필요한 주요 로직을 포함하는 폴더
- `train_runner.py`는 학습을 진행하는 클래스이며, `test_runner.py`는 모델 평가를 수행합니다.
### 7) `utils`
- 학습과 테스트 과정에서 사용되는 유틸리티 함수들을 정의한 폴더
- `utils.py`는 로깅, 체크포인트 저장 등 다양한 기능을 제공합니다.

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
### Main
```bash
python3 main.py
```
- `config.yaml` 파일을 수정한 후, 해당 명령어를 사용하여 학습과 추론을 모두 진행할 수 있습니다. 
### Train
```bash
python tools/train.py configs/a_custom/model_name.py
```
- `--config_path` : 설정 파일 경로 (기본값 : config.yaml)
- `--split_ratio` : 학습/검증 데이터 분할 비율 (기본값 : 0.2)
- `--use_cutmix` : CutMix 사용시 플래그 추가
- `--epochs` : 학습할 에폭 수 (기본값 : 5)
- `--lr` : 학습률 설정
- `--batch_size` : 배치 크기 설정
- `--img_size` : Resize 이미지 크기
- `--model_name` : 사용할 모델 이름 (timm모델 사용, 기본값 : resnet50)

### Test
```bash
python3 test.py --model_name resnet50 --file_path ./best_model.pt
```
- `--model_name` : 모델 아키텍쳐 이름 (필수)
- `--file_path` : 저장된 모델 파일 경로 (필수)

<br />

## 🏅 Wrap-Up Report   
### [ Wrap-Up Report 👑](https://github.com/boostcampaitech7/level1-imageclassification-cv-08/blob/main/warm_up_report/CV%EA%B8%B0%EC%B4%88%EB%8C%80%ED%9A%8C_CV_%ED%8C%80%20%EB%A6%AC%ED%8F%AC%ED%8A%B8(08%EC%A1%B0).pdf)
