_base_ = [
    'configs/_base_/models/upernet_swin.py', 
    'configs/_base_/default_runtime.py'
]

model = dict(
    pretrained='pretrain/swin_base_patch4_window7_224_22k_mm.pth',
    backbone=dict(
        embed_dims=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32]),
    decode_head=dict(in_channels=[128, 256, 512, 1024], num_classes=19),
    auxiliary_head=dict(in_channels=512, num_classes=19))


# dataset settings
dataset_type = 'CustomDataset'
classes = ('background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth')

data_root = '/search/odin/sxf/mmsegmentation/data/celeba_mask/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

img_scale = (512, 512)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
#    dict(type='Resize', img_scale=img_scale, keep_ratio=True),
#    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0),
#    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
#    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
       #img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
        img_scale = img_scale,
        flip=False,
        transforms=[
#            dict(type='Resize', keep_ratio=True),
#            dict(type='RandomFlip', prob=0),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
#        img_dir='img/test',
#        ann_dir='ann/test',
        img_dir='img/train',
        ann_dir='ann/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/val',
        ann_dir='ann/val',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img/test',
        ann_dir='ann/test',
        pipeline=test_pipeline))

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
#    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
#    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

optimizer_config = dict()

checkpoint_config = dict(by_epoch=True, interval=1)

runner = dict(type='EpochBasedRunner', max_epochs=100)
evaluation = dict(interval=1, metric='mIoU')

log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='gloo')


