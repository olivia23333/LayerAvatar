name = 'ssdnerf_avatar_uncond_16bit_composite_new_wobug'

model = dict(
    type='DiffusionNeRF',
    code_size=(12, 128, 128*3),
    code_reshape=(12, 128, 128*3),
    code_activation=dict(
        type='NormalizedTanhCode', mean=0.0, std=0.5, clip_range=2),
    grid_size=64,
    diffusion=dict(
        type='GaussianDiffusion',
        num_timesteps=1000,
        betas_cfg=dict(type='linear'),
        denoising=dict(
            type='DenoisingUnetMod',
            image_size=[128, 128*3],
            in_channels=12,
            base_channels=128,
            channels_cfg=[0.5, 1, 2, 2, 4, 4],
            resblocks_per_downsample=2,
            dropout=0.0,
            use_scale_shift_norm=True,
            downsample_conv=True,
            upsample_conv=True,
            num_heads=4,
            attention_res=[16, 8, 4]),
        timestep_sampler=dict(
            type='SNRWeightedTimeStepSampler',
            power=0.5),
        ddpm_loss=dict(
            type='DDPMMSELossMod',
            rescale_mode='timestep_weight',
            log_cfgs=dict(
                type='quartile', prefix_name='loss_mse', total_timesteps=1000),
            data_info=dict(pred='v_t_pred', target='v_t'),
            weight_scale=20)),
    decoder=dict(
        type='UVTPDecoder',
        interp_mode='bilinear',
        base_layers=[6, 64],
        density_layers=[64, 1],
        color_layers=[6, 128, 3],
        cov_layers=[128, 6],
        offset_layers=[64, 3],
        use_dir_enc=False,
        dir_layers=[16, 64],
        activation='silu',
        bg_color=1,
        # sigma_activation='trunc_exp',
        sigma_activation='sigmoid',
        sigmoid_saturation=0.001,
        gender='neutral',
        max_steps=256,
        multires=0,
        image_size=1024,
        superres=False,
        ),
    decoder_use_ema=True,
    freeze_decoder=False,
    bg_color=1,
    pixel_loss_weight=18,
    seg_loss_weight=18,
    init_seg_weight=5,
    inner_loss_weight=5,
    skin_loss_weight=0.5,
    opacity_loss_weight=0.5,
    per_loss=dict(
        type='PerLoss',
        loss_weight=0.1,
        height=512,
        width=512),
    reg_loss=dict(
        type='TVLoss',
        power=2,
        loss_weight=0.5),
    cache_size=1954,
    scale_loss_weight=0,
    cache_16bit=True)

save_interval = 5000
eval_interval = 5000
code_dir = '/mnt/sdb/zwt/LayerAvatar/cache/' + name + '/code'
work_dir = '/mnt/sdb/zwt/LayerAvatar/work_dirs/' + name
# code_dir = '/home/zhangweitian/HighResAvatar/cache/' + name + '/code'
# work_dir = '/home/zhangweitian/HighResAvatar/work_dirs/' + name

train_cfg = dict(
    dt_gamma_scale=0.5,
    density_thresh=0.005,
    extra_scene_step=3,
    n_inverse_rays=2 ** 12,
    n_decoder_rays=2 ** 12,
    loss_coef=0.1 / (1024 * 1024),
    # optimizer=dict(type='Adam', lr=0.04, weight_decay=0.),
    optimizer=dict(type='Adam', lr=0.04, weight_decay=0.),
    cache_load_from=code_dir,
    viz_dir=None,
    densify_start_iter=20000,
    init_iter=8000,
    offset_weight=5,
    scale_weight=0.0,)
test_cfg = dict(
    img_size=(1024, 1024),
    num_timesteps=50,
    clip_range=[-2, 2],
    density_thresh=0.005,
    # offset_weight=5,
    # scale_weight=0.0,
    # n_inverse_rays=2 ** 20,
    # override_cfg={'diffusion_ema.ddpm_loss.weight_scale': 15.0},
    # loss_coef=0.1 / (1024 * 1024),
    # guidance_gain=0.05 * (2 ** 20),
    # cond_mode='guide_optim',
    # extra_scene_step=3,
    # n_inverse_steps=20,
    # optimizer=dict(type='Adam', lr=0.01, weight_decay=0.),
    # lr_scheduler=dict(type='ExponentialLR', gamma=0.998)
)

optimizer = dict(
    diffusion=dict(type='Adam', lr=1e-4, weight_decay=0.),
    decoder=dict(type='Adam', lr=1e-3, weight_decay=0.))
dataset_type = 'PartDataset'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        data_prefix='data/humanscan_composite/human_train',
        cache_path='data/humanscan_composite/human_train_cache.pkl',
        specific_observation_num=8,
        img_res=1024),
    val_uncond=dict(
        type=dataset_type,
        # data_prefix='data/humanscan_composite/human_train',
        data_prefix='data/humanscan_composite/human_recon_noise3',
        # data_prefix='data/humanscan_composite/human_sv_recon',
        load_imgs=False,
        num_test_imgs=54,
        # num_test_imgs=1,
        scene_id_as_name=True,
        img_res=1024,
        # cache_path='data/humanscan_composite/human_train_cache.pkl',
        cache_path='data/humanscan_composite/human_reconnoise3_cache.pkl',
        # cache_path='data/humanscan_composite/human_svrecon_cache.pkl'
        ),
    val_cond=dict(
        type=dataset_type,
        data_prefix='data/humanscan_composite/human_sv_recon',
        specific_observation_idcs=[0],
        img_res=1024,
        cache_path='data/humanscan_composite/human_svrecon_cache.pkl'),
    train_dataloader=dict(split_data=True))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    gamma=0.5,
    step=[100000])
checkpoint_config = dict(interval=save_interval, by_epoch=False, max_keep_ckpts=2)

evaluation = [
    dict(
        type='GenerativeEvalHook3D',
        data='val_uncond',
        interval=eval_interval,
        feed_batch_size=32,
        viz_step=1,
        metrics=dict(
            type='FIDKID',
            num_images=1954 * 18,
            inception_pkl='work_dirs/cache/composite_train_inception_stylegan.pkl',
            inception_args=dict(
                type='StyleGAN',
                inception_path='work_dirs/cache/inception-2015-12-05.pt'),
            bgr2rgb=False),
        viz_dir=work_dir + '/viz_uncond_vizdress',
        save_best_ckpt=False)]

total_iters = 200000
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
    ])
# yapf:enable

custom_hooks = [
    dict(
        type='ExponentialMovingAverageHook',
        module_keys=('diffusion_ema', 'decoder_ema'),
        interp_mode='lerp',
        interval=1,
        start_iter=0,
        momentum_policy='rampup',
        momentum_cfg=dict(
            ema_kimg=4, ema_rampup=0.05, batch_size=8, eps=1e-8),
        priority='VERY_HIGH'),
    dict(
        type='SaveCacheHook',
        interval=save_interval,
        by_epoch=False,
        out_dir=code_dir,
        viz_dir='cache/' + name + '/viz'),
    dict(
        type='ModelUpdaterHook',
        step=[8000, 50000, 100000],
        cfgs=[{'train_cfg.extra_scene_step': 1},
              {'train_cfg.extra_scene_step': 1,
               'train_cfg.offset_weight': 1,
               'pixel_loss_weight': 10.0,
               'seg_loss_weight':10.0,
            #    'scale_loss_weight':0.5,
               'inner_loss_weight':2.5,
               'skin_loss_weight':0.25,
               'opacity_loss_weight':0.25,
               'reg_loss.loss_weight':0.25,
               'per_loss.loss_weight':0.05,},
              {'train_cfg.extra_scene_step': 1,
               'train_cfg.optimizer.lr': 0.01,
               'train_cfg.offset_weight': 0.5,
               'pixel_loss_weight': 5.0,
               'seg_loss_weight':5.0,
            #    'scale_loss_weight':0.25,
               'inner_loss_weight':1.25,
               'skin_loss_weight':0.125,
               'opacity_loss_weight':0.05,
               'reg_loss.loss_weight':0.125,
               'per_loss.loss_weight':0.025,}],
        by_epoch=False)
]

# use dynamic runner
runner = dict(
    type='DynamicIterBasedRunner',
    is_dynamic_ddp=False,
    pass_training_status=True)
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', save_interval)]
use_ddp_wrapper = True
find_unused_parameters = False
cudnn_benchmark = True
opencv_num_threads = 0
mp_start_method = 'fork'