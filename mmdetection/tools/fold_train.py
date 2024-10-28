# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner

from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()
    data_root = '/data/ephemeral/home/dataset/'
    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)

    # 전체 5개 폴드에 대해 순차적으로 학습
    for fold_idx in range(5):
        print(f"Training fold {fold_idx}...")
        
        # 폴드 인덱스에 맞는 train/val 데이터 경로 설정
        cfg.train_dataloader.dataset.ann_file = f'stfold/train_kfold_{fold_idx}.json'
        cfg.val_dataloader.dataset.ann_file = f'stfold/val_kfold_{fold_idx}.json'
        cfg.val_evaluator.ann_file=data_root + f'stfold/val_kfold_{fold_idx}.json'

        # # MLflow 관련 설정에서 run_name 동적으로 변경
        # for vis_backend in cfg.visualizer['vis_backends']:
        #     if vis_backend['type'] == 'MLflowVisBackend':
        #         vis_backend['run_name'] = f'fold_{fold_idx}_run'
        
        # work_dir도 각 폴드별로 저장 위치 다르게 설정
        fold_work_dir = osp.join('./work_dirs', f'fold_{fold_idx}')
        cfg.work_dir = fold_work_dir

        # 폴드별로 로그를 저장할 수 있도록 work_dir 생성
        os.makedirs(cfg.work_dir, exist_ok=True)

        # enable automatic-mixed-precision training
        if args.amp:
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

        # enable automatically scaling LR
        if args.auto_scale_lr:
            if 'auto_scale_lr' in cfg and \
                    'enable' in cfg.auto_scale_lr and \
                    'base_batch_size' in cfg.auto_scale_lr:
                cfg.auto_scale_lr.enable = True
            else:
                raise RuntimeError('Cannot find "auto_scale_lr" or '
                                '"auto_scale_lr.enable" or '
                                '"auto_scale_lr.base_batch_size" in your'
                                ' configuration file.')

        # resume is determined in this priority: resume from > auto_resume
        if args.resume == 'auto':
            cfg.resume = True
            cfg.load_from = None
        elif args.resume is not None:
            cfg.resume = True
            cfg.load_from = args.resume

        # build the runner from config
        if 'runner_type' not in cfg:
            # build the default runner
            runner = Runner.from_cfg(cfg)
        else:
            # build customized runner from the registry
            runner = RUNNERS.build(cfg)

        # 폴드별로 학습 시작
        runner.train()

if __name__ == '__main__':
    main()
