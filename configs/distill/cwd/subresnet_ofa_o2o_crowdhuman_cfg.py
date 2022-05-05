f=256
teacher_checkpoint='/media/Pochinki/mmdetection/experiments/ofa_o2o_crowdhuman_cfg_with_eval/epoch_40.pth'
teacher = dict(
    type='mmdet.FCOSO2O',
    init_cfg=dict(type='Pretrained', checkpoint=teacher_checkpoint),
    backbone=dict(
        type='ResNet50D',
    #checkpoint_path='/media/Pochinki/mmdetection/EXP_OFA/train-ResNet50D-G_0-CMP_0-pretraining_Resnet50D_for_counting_NAS-EXP_OFA-cifar10-20220325-014407/checkpoint.pth.tar',
    bn_param=(0.1, 1e-5),
        dropout_rate=0,
        width_mult=1.0,
        depth_param=3,
        expand_ratio=0.35
    ),
        #depth=50,
        #num_stages=4,
        #out_indices=(0, 1, 2, 3),
        #frozen_stages=1,
        #norm_cfg=dict(type='BN', requires_grad=True),
        #norm_eval=True,
        #style='pytorch',
        #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='O2OHead',
        num_classes=1,
        in_channels=256,
        top_channels=256,
        num_top_layers=4,
        top_norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(
        assigner=dict(type='ATSSMaskAssigner', ctr_type='mask', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        max_proposals=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=500))
student_checkpoint='/media/Pochinki/mmdetection/experiments/subresnet_ofa_o2o_crowdhuman_cfg_with_eval/epoch_40.pth'
student = dict(
    type='mmdet.FCOSO2O',
    init_cfg=dict(type='Pretrained', checkpoint=student_checkpoint),
    backbone=dict(
        type='SubResNet',
        d_list=[2,0,0,0,0],
        e_list=[0.2, 0.2, 0.2, 0.35, 0.35, 0.2, 0.2, 0.25, 0.35,
                0.2, 0.35, 0.2, 0.25, 0.25, 0.2, 0.2],
        w_list=[0, 1, 1, 0, 0, 0],),
        #checkpoint_path='/media/Pochinki/mmdetection/EXP_OFA/train-subresnet-G_0-CMP_0-pretraining_subresnet_for_counting_NAS-EXP_OFA-cifar10-20220325-020740/checkpoint.pth.tar'),
        #depth=50,
        #num_stages=4,
        #out_indices=(0, 1, 2, 3),
        #frozen_stages=1,
        #norm_cfg=dict(type='BN', requires_grad=True),
        #norm_eval=True,
        #style='pytorch',
        #init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        #in_channels=[256, 512, 1024, 2048],
        in_channels=[128, 152, 304, 616],
        out_channels=f, #224, #192, #256
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='O2OHead',
        num_classes=1,
        in_channels=f,#224, #192, #256,
        top_channels=f,#224, #192, #256,
        num_top_layers=4,
        top_norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(
        assigner=dict(type='ATSSMaskAssigner', ctr_type='mask', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        max_proposals=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=500))
# algorithm setting
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='bbox_head.conv_cls',#'bbox_head.gfl_cls',
                teacher_module='bbox_head.conv_cls',#'bbox_head.gfl_cls',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_cls_head',
                        tau=1,
                        loss_weight=1,
                    )
                ]),
            dict(
                student_module='bbox_head.conv_reg',
                teacher_module='bbox_head.conv_reg',
                losses=[
                    dict(
                        type='ChannelWiseDivergence',
                        name='loss_cwd_reg_head',
                        tau=1,
                        loss_weight=1,
                    )
                ])
        ]),
)
optimizer = dict(type='AdamW', lr=4e-05, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=1, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=0.01,
    step=[30])
dataset_type = 'CrowdHumanDataset'
data_root = '/media/Pochinki/mmdetection/data/head_detection/'
img_scale = [(1400, 896), (1400, 864), (1400, 832), (1400, 800), (1400, 768),
             (1400, 736), (1400, 704), (1400, 672), (1400, 640)]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(1400, 896), (1400, 864), (1400, 832), (1400, 800),
                   (1400, 768), (1400, 736), (1400, 704), (1400, 672),
                   (1400, 640)],
        multiscale_mode='value',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1400, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='CrowdHumanDataset',
        ann_file='/media/Pochinki/mmdetection/data/head_detection/annotations/all_train.json',
        img_prefix='/media/Pochinki/mmdetection/data/head_detection/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Resize',
                img_scale=[(1400, 896), (1400, 864), (1400, 832), (1400, 800),
                           (1400, 768), (1400, 736), (1400, 704), (1400, 672),
                           (1400, 640)],
                multiscale_mode='value',
                keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]),
    val=dict(
        type='CrowdHumanDataset',
        ann_file='/media/Pochinki/mmdetection/data/head_detection/annotations/all_train.json',
        img_prefix='/media/Pochinki/mmdetection/data/head_detection/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1400, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CrowdHumanDataset',
        ann_file='/media/Pochinki/mmdetection/data/head_detection/annotations/all_train.json',
        img_prefix='/media/Pochinki/mmdetection/data/head_detection/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(1400, 800),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=10, metric=['bbox'])
work_dir = 'experiments/dis_{}_subresnet_ofa_o2o_crowdhuman_cfg_with_eval_pretrained_cls1_reg1/'.format(f)
checkpoint_config = dict(interval=10)
# log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
log_config = dict(interval=1,
                  hooks=[dict(type='NeptuneLoggerHook',
                              init_kwargs=dict(project='uzair789/mmdetection',
                                               name='cwd_{}_subresnet_ofa_o2o_crowdhuman_cfg_with_eval_pretrained_cls1_reg1'.format(f))
                              )
                         ]
                  )
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
runner = dict(type='EpochBasedRunner', max_epochs=40)
gpu_ids = range(0, 8)
auto_resume = False
find_unused_parameters=True
