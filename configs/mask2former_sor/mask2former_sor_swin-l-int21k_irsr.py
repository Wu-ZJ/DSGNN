_base_ = ['./mask2former_sor_swin-b_irsr.py']
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

model = dict(
    backbone=dict(
        embed_dims=192,
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    panoptic_head=dict(
        num_queries=8,
        in_channels=[192, 384, 768, 1536],
        score_thr=0.75))

data = dict(samples_per_gpu=2, workers_per_gpu=2)

lr_config = dict(step=[15000, 30000])
max_iters = 36000
runner = dict(type='IterBasedRunner', max_iters=max_iters)

interval = 1000
workflow = [('train', interval)]
checkpoint_config = dict(
    by_epoch=False, interval=interval, save_last=True, max_keep_ckpts=30)
dynamic_intervals = [(max_iters // interval * interval + 1, max_iters)]
evaluation = dict(
    interval=960000,
    dynamic_intervals=dynamic_intervals,
    metric=['mae'])