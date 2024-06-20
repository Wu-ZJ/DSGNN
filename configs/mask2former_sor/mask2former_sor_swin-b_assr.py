_base_ = ['./mask2former_sor_swin-t_assr.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window12_384.pth'  # noqa

depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        pretrain_img_size=384,
        embed_dims=128,
        depths=depths,
        num_heads=[4, 8, 16, 32],
        window_size=12,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(in_channels=[128, 256, 512, 1024]))

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
    'structure_query_embed': embed_multi,
    'structure_query_feat': embed_multi,
    'texture_query_embed': embed_multi,
    'texture_query_feat': embed_multi,
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

custom_keys.update(
    {'rank_head': dict(lr_mult=1.0, decay_mult=0.0)}
)

# optimizer
optimizer = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
optimizer_config = dict(grad_clip=dict(max_norm=0.01, norm_type=2))
