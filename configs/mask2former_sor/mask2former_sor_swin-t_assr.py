_base_ = ['./mask2former_sor_r50_assr.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
depths = [2, 2, 6, 2]
model = dict(
    type='Mask2Former',
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=depths,
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(
        in_channels=[96, 192, 384, 768],
        score_thr=0.75),
    init_cfg=None)

# dataset settings
image_size = (640, 640)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
pad_cfg = dict(img=(128, 128, 128), masks=0, seg=255)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='SORLoadAnnotations', with_bbox=True, with_mask=True, with_rank_label=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    # large scale jittering
    dict(
        type='Resize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-5, 1e-5), by_mask=True),
    dict(type='Pad', size=image_size, pad_val=pad_cfg),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle', img_to_float=True),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks', 'rank_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Pad', size_divisor=32, pad_val=pad_cfg),
            # dict(type='Pad', size=image_size, pad_val=pad_cfg),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

dataset_type = 'SORDataset'
data_root = '/opt/data/private/SOR/data/ASSR/'

data = dict(
    _delete_=True,
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'obj_seg_data_train_val.json',
        img_prefix=data_root + 'images/train_val/',
        rank_prefix=data_root + 'rank_order/train_val/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'obj_seg_data_val.json',
        img_prefix=data_root + 'images/val/',
        rank_prefix=data_root + 'rank_order/val/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'obj_seg_data_test.json',
        img_prefix=data_root + 'images/test/',
        rank_prefix=data_root + 'rank_order/test/',
        pipeline=test_pipeline))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    weight_decay=0.05,
    eps=1e-8,
    betas=(0.9, 0.999),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))