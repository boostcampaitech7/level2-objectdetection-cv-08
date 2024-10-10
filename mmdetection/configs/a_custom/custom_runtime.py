default_scope = 'mmdet'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# LOCAL에 저장
vis_backends = [dict(type='LocalVisBackend')]

# Visualizer에 MLflow 연결
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='MLflowVisBackend',
        save_dir='/data/ephemeral/home/db_dir',
        exp_name='recycle_detection_experiment',
        run_name=f'fold_run',
        tracking_uri='https://f0bf-223-130-141-5.ngrok-free.app',
        artifact_suffix=['.json', '.log', '.py', 'yaml']
    )
]

visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# 로그 설정
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
