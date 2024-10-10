import mlflow
import os
import torch
from mmcv import Config
from mmdet.apis import train_detector, set_random_seed, init_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv.runner import get_dist_info, init_dist

def main():
    # MLflow 설정
    mlflow.set_tracking_uri("https://your_mlflow_server.com")  # MLflow 서버 URL
    mlflow.set_experiment("detr_experiment")  # 원하는 실험명 설정

    with mlflow.start_run(run_name="detr_training"):
        # 설정 파일 로드 (mmdetection config)
        cfg = Config.fromfile('configs/detr/detr_custom.py')
        
        # 여기서 필요한 파라미터를 MLflow에 기록
        mlflow.log_param("epochs", cfg.runner.max_epochs)
        mlflow.log_param("learning_rate", cfg.optimizer.lr)
        mlflow.log_param("batch_size", cfg.data.samples_per_gpu)

        # 로그 폴더 생성
        cfg.work_dir = './work_dirs/detr_custom_with_mlflow'
        os.makedirs(cfg.work_dir, exist_ok=True)

        # 모델과 데이터셋 빌드
        datasets = [build_dataset(cfg.data.train)]
        model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))

        # 학습 중간에 랜덤 시드 설정 (선택 사항)
        if cfg.get('seed') is not None:
            set_random_seed(cfg.seed, deterministic=True)
        
        # GPU 사용 설정
        cfg.gpu_ids = range(1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 학습 시작
        train_detector(model, datasets, cfg, distributed=False, validate=True)

        # 학습이 끝난 후 최종 체크포인트와 결과를 MLflow에 기록
        best_checkpoint = os.path.join(cfg.work_dir, 'latest.pth')
        mlflow.log_artifact(best_checkpoint)  # 모델 가중치 기록
        mlflow.log_artifact(os.path.join(cfg.work_dir, 'train.log'))  # 로그 파일 기록

if __name__ == '__main__':
    main()
