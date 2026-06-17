custom_imports = dict(
    imports=[
        'custom_datasets.customAgri',
        'custom_models.dinov3_backbone_fb',
    ],
    allow_failed_imports=False,
)

_base_ = [
    '/mnt/ht2-nas2/00-model/00-fb/MMcodes/mmsegmentation/configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py',
]
# ── Paths ─────────────────────────────────────────────────────────────────────
DINO_CKPT   = '/mnt/ht2-nas2/00-model/00-fb/mmseg_data/weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
DATA_ROOT   = '/mnt/ht2-nas2/00-model/00-jiangzf/label20000/Segmentation/' 
# ──────────────────────────────────────────────────────────────────────────────

img_size    = 256
num_classes = 2

# 因为我们已经在 LoadCustomRaster 里归一化了，所以必须删掉原本的预处理器归一化参数，
# 只保留 GPU 的 padding (Pad_val) 保证 Mask2Former 不报错。
data_preprocessor = dict(
    type='SegDataPreProcessor',
    _delete_=True,
    mean=None,  
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255, # 255 是忽略索引
    size=(img_size, img_size),
    test_cfg=dict(size_divisor=32),
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='DINOv3BackboneMmseg_fb',
        arch='vit_large',
        patch_size=16,
        checkpoint=DINO_CKPT,
        freeze_backbone=False,
    ),
    decode_head=dict(
        in_channels=[1024, 1024, 1024, 1024],
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        loss_cls=dict(
            _delete_=True, # 强行覆盖原版可能存在的 FocalLoss 字典
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            # class_weight=[1.0] * num_classes + [0.1],
            class_weight=[1.0,2.0,0.1],
        ),
    ),
)

# ── 你的专属数据流水线 (含归一化配置) ─────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadCustomRaster', img_size=img_size),
    dict(type='CustomRandomRotate90', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadCustomRaster', img_size=img_size),
    dict(type='PackSegInputs'),
]

# ── 数据集加载器 (使用完全自定义的 split 逻辑) ────────────────────────────────────
train_dataloader = dict(
    _delete_=True,  # 确保原有的所有加载策略被抹除
    batch_size=16,
    num_workers=4,
    dataset=dict(
        type='CustomAgriDataset',
        data_root=DATA_ROOT,
        split='train',  # 代码里会自动去找 train.txt
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='CustomAgriDataset',
        data_root=DATA_ROOT,
        split='valid',  # 代码里会自动去找 valid.txt
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=4,
    num_workers=4,
    dataset=dict(
        type='CustomAgriDataset',
        data_root=DATA_ROOT,
        # 注意: 如果你的测试集文件名为 test10.txt，请改为 split='test10'
        split='test10',   
        pipeline=val_pipeline,
    ),
)

val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU'], _delete_=True)
test_evaluator = val_evaluator

# ── Optimiser 和 Schedule ──────────────────────────────────────────────────────
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05,
                   eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            # 修复：仅对ViT本身（预训练权重）缩小学习率，保证Adapter正常收敛
            'backbone.adapter.backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat':  embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-3, begin=0, end=1500, by_epoch=False),
    dict(type='PolyLR', eta_min=0, power=0.9, begin=1500, end=40000, by_epoch=False),
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)
val_cfg   = dict(type='ValLoop')
test_cfg  = dict(type='TestLoop')

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False,
                    interval=2000, save_best='mIoU', max_keep_ckpts=1),
    logger=dict(type='LoggerHook', interval=100, log_metric_by_epoch=False),
)

find_unused_parameters = True