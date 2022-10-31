# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    '../_base_/default_runtime.py',
    # DAFormer Network Architecture
    '../_base_/models/deeplabv3plus_r50-d8.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_potsdam_to_vaihingen_512x512.py',
    # Basic UDA Self-Training
    '../_base_/uda/fmda_mixed.py',
    # AdamW Optimizer
    '../_base_/schedules/adamw.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/poly10warm.py'
]
# Random Seed
seed = 0

model = dict(decode_head=dict(num_classes=6))

# Modifications to Basic UDA
uda = dict(
    # Increased Alpha
    alpha=0.999,
    # Thing-Class Feature Distance
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=[1, 3, 4, 5],
    imnet_feature_dist_scale_min_ratio=0.75,
    # Pseudo-Label Crop
    pseudo_weight_ignore_top=15,
    pseudo_weight_ignore_bottom=120,
    aux_losses=[
        dict(
            type='AdaptiveFeatSimLoss',
            kernel_size=3,
            dilation=4,
            top_k=3,
            sigma=35,
            weights={'src_pos': 0.1, 'src_neg': 0.1, 'sim_pos': 0.1, 'sim_neg': 0.1},
            sim_type='cosine',
            feat_level=2
        ),
    ],
    trg_loss_weight=1.
)
data = dict(
    train=dict(
        # Rare Class Sampling
        # rare_class_sampling=dict(
        #     min_pixels=3000, class_temp=0.01, min_crop_ratio=0.5)
        rare_class_sampling=None
    )
)
# Optimizer Hyperparameters
optimizer_config = None
optimizer = dict(
    lr=6e-05,
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            pos_block=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
n_gpus = 1
runner = dict(type='IterBasedRunner', max_iters=40000)
# Logging Configuration
checkpoint_config = dict(by_epoch=False, interval=4000, max_keep_ckpts=10)
evaluation = dict(interval=4000, metric='mIoU')
# Meta Information for Result Analysis
name = 'fmda_mixed_deeplabv3plus_r50_warm_croppl_a999_ada_potsdam2vaihingen'
exp = 'basic'
name_dataset = 'potsdam2vaihingen'
name_architecture = 'fmda_mixed_deeplabv3plus_r50'
name_encoder = 'r50'
name_decoder = 'deeplabv3plus'
name_uda = 'dacs_a999_fd_things_cpl'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'

init_kwargs = dict(
    project='rsi_segmentation',
    entity='tum-tanmlh',
    name=name,
    resume='never'
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook'),
        dict(type='WandbHookSeg',
             init_kwargs=init_kwargs,
             interval=201),
        dict(type='MMSegWandbHook',
             init_kwargs=init_kwargs,
             interval=201,
             num_eval_images=0),
        dict(type='PlotStatisticsHook',
             log_dir=f'work_dirs/plots/{name}',
             sim_feat_cfg=dict(kernel_size=3, dilation=2,
                               sigma=21.41786553,
                               feat_level=3,
                               mean_sim=[0.5, 0.55, 0.6, 0.65, 0.7],
                               top_k=9),
             data_cfg=data,
             interval=1),
        # dict(type='PseudoLabelingHook',
        #      log_dir='work_dirs/pseudo_labels/deeplabv3plus_r50-d8_512x512_80k_loveda_r2u',
        #      interval=1),
    ])
