log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

map_classes =['Diversion']
fixed_ptsnum_per_gt_line = 50 # now only support fixed_pts > 0
fixed_ptsnum_per_pred_line = 50

eval_use_same_gt_sample_num_flag=False
num_map_classes =len(map_classes)


_dim_ = 256
_ffn_dim_ = _dim_*2
instance_num=50
model = dict(
    type='VMA',
    use_grid_mask=True,
    video_test_mode=False,
    pretrained=dict(img='ckpts/resnet152-b121ed2d.pth'),
    img_backbone=dict(
        type='iCurb_BackBone',
        out_indice=(1, 2, 3,)),
    img_neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    pts_bbox_head=dict(
        type='VMAHead',
        num_query=900,
        num_vec=instance_num,
        num_pts_per_vec=fixed_ptsnum_per_pred_line, # one line
        num_pts_per_gt_vec=fixed_ptsnum_per_gt_line,
        dir_interval=1,
        query_embed_type='instance_pts',
        transform_method='minmax',
        num_classes=num_map_classes,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        code_size=2,
        code_weights=[1.0, 1.0, 1.0, 1.0],
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='VMADetectionTransformerDecoder',
                num_layers=7,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_),
                    ],

                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='VMANMSFreeCoder',
            max_num=instance_num,
            score_threshold=0.3,
            num_classes=num_map_classes),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_attr=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.0),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0),
        loss_pts2lines=dict(type='Pts2LinesLoss', 
                      loss_weight=0.0),
        loss_pts2pts=dict(type='PtsL1Loss', 
                      loss_weight=10.0),
        loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.05)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        out_size_factor=4,
        assigner=dict(
            type='VMAHungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=0.0, box_format='xywh'),
            # reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            # iou_cost=dict(type='IoUCost', weight=1.0), # Fake cost. This is just to make it compatible with DETR head.
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=0.0),
            pts_cost=dict(type='OrderedPtsL1Cost', 
                      weight=5),
            )
        )
    )
)

pipeline = [
    dict(type='LoadImageFromFiles', to_float32=True),
    dict(type='PadChannel'),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['img_shape', 'filename'],
    )
]

dataset_type = 'SD_Driving_Freespace_Dataset'
data_root = 'data/sd_data/freespace/cropped_data'
file_client_args = dict(backend='disk')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        sub_dir = 'trajectory_cropped_images', 
        annotation_file = 'sd_data_freespace_dict.json',
        mask_dir = data_root + '/mask_map',
        points_nums=fixed_ptsnum_per_gt_line,
        map_classes=map_classes, 
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        pipeline=pipeline,
        mode='train',
        test_mode=False),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        sub_dir = 'trajectory_cropped_images', 
        annotation_file = 'sd_data_freespace_dict.json',
        mask_dir = data_root + '/mask_map',
        points_nums=fixed_ptsnum_per_gt_line,
        map_classes=map_classes,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        pipeline=pipeline,
        mode='valid',
        test_mode=False),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        sub_dir = 'trajectory_cropped_images', 
        annotation_file = 'sd_data_freespace_dict.json',
        mask_dir = data_root + '/mask_map',
        points_nums=fixed_ptsnum_per_gt_line,
        map_classes=map_classes,
        eval_use_same_gt_sample_num_flag=eval_use_same_gt_sample_num_flag,
        pipeline=pipeline,
        mode='test',
        test_mode=True),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=7.5e-5,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)

total_epochs = 80
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=5)

evaluation = dict(interval=80, metric='chamfer')
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
fp16 = dict(loss_scale=512.)
load_from = 'ckpts/seg_pretrain.pth'
