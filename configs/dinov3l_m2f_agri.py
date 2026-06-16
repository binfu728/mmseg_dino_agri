_base_ = [
    '../mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py',
]

custom_imports = dict(imports=['custom_dataset'], allow_failed_imports=False)

# ── Paths ─────────────────────────────────────────────────────────────────────
DINO_CKPT   = '/home/zifei/.cache/modelscope/hub/models/facebook/dinov3pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
DATA_ROOT   = '/path/to/your/custom/dataset/folder' 
# ──────────────────────────────────────────────────────────────────────────────

img_size    = 512
num_classes = 2

# 数据集的均值和方差，我们现在直接传给 CPU Pipeline
DATA_MEAN = [72.4085, 89.7399, 69.6123]
DATA_STD  = [32.8544, 23.9954, 23.1234]

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
        type='DINOv3BackboneMmseg',
        arch='vit_small',
        patch_size=16,
        checkpoint=DINO_CKPT,
        freeze_backbone=True,
    ),
    decode_head=dict(
        in_channels=[384, 384, 384, 384],
        strides=[4, 8, 16, 32],
        num_classes=num_classes,
        loss_cls=dict(
            _delete_=True, # 强行覆盖原版可能存在的 FocalLoss 字典
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 1.0, 0.1],
        ),
    ),
)

# ── 你的专属数据流水线 (含归一化配置) ─────────────────────────────────────────────
train_pipeline = [
    dict(type='LoadCustomRaster', img_size=img_size, mean=DATA_MEAN, std=DATA_STD),
    dict(type='CustomRandomRotate90', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

val_pipeline = [
    dict(type='LoadCustomRaster', img_size=img_size, mean=DATA_MEAN, std=DATA_STD),
    dict(type='PackSegInputs'),
]

# ── 数据集加载器 (使用完全自定义的 split 逻辑) ────────────────────────────────────
train_dataloader = dict(
    _delete_=True,  # 确保原有的所有加载策略被抹除
    batch_size=4,
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
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CustomAgriDataset',
        data_root=DATA_ROOT,
        split='valid',  # 代码里会自动去找 valid.txt
        pipeline=val_pipeline,
    ),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=2,
    dataset=dict(
        type='CustomAgriDataset',
        data_root=DATA_ROOT,
        # 注意: 如果你的测试集文件名为 test10_txt.txt，请改为 split='test10_txt'
        split='test',   
        pipeline=val_pipeline,
    ),
)

val_evaluator  = dict(type='IoUMetric', iou_metrics=['mIoU'], _delete_=True)
test_evaluator = val_evaluator

# ── Optimiser 和 Schedule ──────────────────────────────────────────────────────
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)

optim_wrapper = dict(
    _delete_=True, # 删除原版 ADE20K 绑定的优化器，防止出现冲突
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': embed_multi,
            'query_feat':  embed_multi,
            'level_embed': embed_multi,
        },
        norm_decay_mult=0.0,
    ),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=2000, _delete_=True)
val_cfg   = dict(type='ValLoop', _delete_=True)
test_cfg  = dict(type='TestLoop', _delete_=True)

# 调度器直接覆写，Scheduler 是列表类型，通常直接给一个新的就行，但由于底层的合并机制可能会追加，
# 建议通过 _delete_ 删除父类存在的 Scheduler 列表（字典内操作）
param_scheduler = [
    dict(type='PolyLR', eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False),
]

default_hooks = dict(
    _delete_=True,
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, save_best='mIoU'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)