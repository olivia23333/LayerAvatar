import torch
import torch.nn as nn
import mmcv

from copy import deepcopy
import imageio
import os
import time
import numpy as np
from torch.nn.parallel.distributed import DistributedDataParallel
# from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
from mmgen.models.builder import MODELS, build_module
from mmgen.models.architectures.common import get_module_device

from ...core import eval_psnr, rgetattr, module_requires_grad, get_cam_rays
from .multiscene_nerf import MultiSceneNeRF


@MODELS.register_module()
class DiffusionNeRF_S(MultiSceneNeRF):

    def __init__(self,
                 *args,
                 diffusion=dict(type='GaussianDiffusion'),
                 diffusion_use_ema=True,
                 freeze_decoder=True,
                 image_cond=False,
                 code_permute=None,
                 code_reshape=None,
                 autocast_dtype=None,
                 ortho=True,
                 return_norm=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        diffusion.update(train_cfg=self.train_cfg, test_cfg=self.test_cfg)
        # torch.autograd.set_detect_anomaly(True)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            self.diffusion_ema = deepcopy(self.diffusion)
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            self.decoder.requires_grad_(False)
            if self.decoder_use_ema:
                self.decoder_ema.requires_grad_(False)
        self.image_cond = image_cond
        self.code_permute = code_permute
        self.code_reshape = code_reshape
        self.code_reshape_inv = [self.code_size[axis] for axis in self.code_permute] if code_permute is not None \
            else self.code_size
        self.code_permute_inv = [self.code_permute.index(axis) for axis in range(len(self.code_permute))] \
            if code_permute is not None else None

        self.autocast_dtype = autocast_dtype
        self.ortho = ortho
        self.return_norm = return_norm
        # zero_code = torch.zeros(self.code_size).unsqueeze(0)
        # opacity, offset, rot, rgb, radius = zero_code.split([3, 9, 9, 9, 9], dim=1)
        # self.opacity = nn.Parameter(opacity, requires_grad=True)
        # self.offset = nn.Parameter(offset, requires_grad=True)
        # self.rot = nn.Parameter(rot, requires_grad=True)
        # self.color = nn.Parameter(rgb, requires_grad=True)
        # self.radius = nn.Parameter(radius, requires_grad=True)
        # self.attr = [self.opacity, self.offset, self.rot, self.color, self.radius]

        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key)

    def code_diff_pr(self, code):
        code_diff = code
        if self.code_permute is not None:
            code_diff = code_diff.permute([0] + [axis + 1 for axis in self.code_permute])  # add batch dimension
        if self.code_reshape is not None:
            code_diff = code_diff.reshape(code.size(0), *self.code_reshape)  # add batch dimension
        return code_diff

    def code_diff_pr_inv(self, code_diff):
        code = code_diff
        if self.code_reshape is not None:
            code = code.reshape(code.size(0), *self.code_reshape_inv)
        if self.code_permute_inv is not None:
            code = code.permute([0] + [axis + 1 for axis in self.code_permute_inv])
        return code

    def train_step(self, data, optimizer, running_status=None):
        iter = running_status['iteration']
        # if optimizer is None:
        #     optimizer = {}
        diffusion = self.diffusion
        decoder = self.decoder_ema if self.freeze_decoder and self.decoder_use_ema else self.decoder
        num_scenes = len(data['scene_id']) # 8
        extra_scene_step = self.train_cfg.get('extra_scene_step', 0) # 15
        
        # start_time = time.time()
        if 'optimizer' in self.train_cfg:
            # code_list_, code_optimizers = self.load_cache(data, self.decoder.num_init)
            # try:
            # start_time = time.time()
            code_list_, code_optimizers = self.load_cache(data)
            # code_list_, code_optimizers, density_grid, density_bitfield = self.load_cache(data)
            code = self.code_activation(torch.stack(code_list_, dim=0), update_stats=True)
            # code = [self.code_activation(attr) for attr in self.attr]
        else:
            assert 'code' in data
            # code, density_grid, density_bitfield = self.load_scene(
            #     data, load_density='decoder' in optimizer)
            # code = self.load_scene(
            #     data, load_density='decoder' in optimizer)
            code = self.load_scene(data, load_density='decoder' in optimizer)
            code_optimizers = []

        # torch.cuda.synchronize()
        # load_time = time.time()
        # print(f'time:load_cache:{load_time-start_time}')
        # if (iter%2) == 0:
        for key in optimizer.keys():
            if key.startswith('diffusion'):
                optimizer[key].zero_grad()
        if 'decoder' in optimizer:
            optimizer['decoder'].zero_grad()
                
        for code_optimizer in code_optimizers:
            code_optimizer.zero_grad()
        # if 'decoder' in optimizer:
        #     optimizer['decoder'].zero_grad()
        # for key in optimizer.keys():
        #     optimizer[key].zero_grad()
        
        concat_cond = None
        if 'cond_imgs' in data:
            cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
            cond_segs = data['cond_segs']
            cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            cond_poses = data['cond_poses'] # (num_scenes, num_imgs, 4, 4)
            smpl_params = data['cond_smpl_param']
            if 'cond_norm' in data:
                cond_norm = data['cond_norm']
            else:
                cond_norm = None

            num_scenes, num_imgs, h, w, _ = cond_imgs.size() # (8, 50, 128, 128)
            # select_ind = torch.randint(num_imgs, (num_scenes,))
            # select_intrinsics = cond_intrinsics[range(num_scenes), select_ind].unsqueeze(1)
            # select_poses = cond_poses[range(num_scenes), select_ind].unsqueeze(1)

            # cameras = [[cond_intrinsics[i, [select_ind[i]]], cond_poses[i, [select_ind[i]]]] for i in range(num_scenes)]
            # target_imgs = cond_imgs[range(num_scenes), select_ind].unsqueeze(1)
            cameras = torch.cat([cond_intrinsics, cond_poses.reshape(num_scenes, num_imgs, -1)], dim=-1)
            # cameras = [[cond_intrinsics[i], cond_poses[i]] for i in range(num_scenes)]
            target_imgs = cond_imgs
            target_segs = cond_segs
            # assert False
            # (num_scenes, num_imgs, h, w, 3)
            # cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
            # R, T = look_at_view_transform(dist=2.6, elev=0, azim=72)
            # print(T)
            # print(R)
            # print(cond_poses[0,4,:3, 3])
            # print(cond_poses[0,4,:3,:3])
            # assert False

            # these two lines are cameras for point rendering
            # cond_poses[:,:,:,:2] *= (-1)
            # cameras = [FoVPerspectiveCameras(fov=46.2, R=cond_poses[i,:,:3,:3], T=cond_poses[i,0,:3, 3].expand(num_imgs, -1), device=cond_poses.device) for i in range(num_scenes)]
            # assert False
            # print(cond_poses.gather(1, select_ind[None, :, None, None]).shape)
            # assert False
            # cameras = [[cond_intrinsics[i, idx], cond_poses[i, idx]] for i, idx in enumerate(select_ind)]
            # cameras = [cond_poses[i, :] for i in range(num_scenes)]
            # assert False
            # cameras = FoVPerspectiveCameras(fov=60, R=cond_poses[:,0:1,:3,:3].flatten(0, 1), T=cond_poses[:,0:1,:3, 3].flatten(0, 1), device=cond_poses.device)
            # dt_gamma_scale = self.train_cfg.get('dt_gamma_scale', 0.0)
            # (num_scenes,)
            # dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))
            dt_gamma = 0.0
            
            if self.image_cond:
                cond_inds = torch.randint(num_imgs, size=(num_scenes,))  # (num_scenes,)
                concat_cond = cond_imgs[range(num_scenes), cond_inds].permute(0, 3, 1, 2)  # (num_scenes, 3, h, w)
                diff_image_size = rgetattr(diffusion, 'denoising.image_size')
                assert diff_image_size[0] % concat_cond.size(-2) == 0
                assert diff_image_size[1] % concat_cond.size(-1) == 0
                concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                                diff_image_size[1] // concat_cond.size(-1)))

        x_t_detach = self.train_cfg.get('x_t_detach', False)
        # torch.cuda.synchronize()
        # init_time = time.time()
        # print(f'init_time:{init_time-load_time}')
        # code shape [8, 3, 6, 128, 128]

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            loss_diffusion, log_vars = diffusion(
                self.code_diff_pr(code), concat_cond=concat_cond, return_loss=True,
                x_t_detach=x_t_detach, cfg=self.train_cfg)
        loss_diffusion.backward()
        # if (iter%2) == 1:
        for key in optimizer.keys():
            if key.startswith('diffusion'):
                optimizer[key].step()
        # torch.cuda.synchronize()
        # diffuse_time = time.time()
        # print(f'diffuse_time:{diffuse_time-init_time}')

        # extra_scene_step = 0
        # log_vars = {}
        if extra_scene_step > 0:
            assert len(code_optimizers) > 0
            prior_grad = [code_.grad.data.clone() for code_ in code_list_]
            cfg = self.train_cfg.copy()
            cfg['n_inverse_steps'] = extra_scene_step
            code, loss_decoder, loss_dict_decoder = self.inverse_code_s(
                decoder, target_imgs, target_segs, cameras, dt_gamma=dt_gamma, smpl_params=smpl_params, cfg=cfg,
                code_=code_list_,
                code_optimizer=code_optimizers,
                prior_grad=prior_grad,
                densify=(iter > cfg['densify_start_iter']),
                init=(iter < cfg['init_iter']),
                cond_norm=cond_norm)
            for k, v in loss_dict_decoder.items():
                log_vars.update({k: float(v)})
        else:
            cfg = self.train_cfg.copy()
            prior_grad = [code_.grad.data.clone() for code_ in code_list_]
            # prior_grad = None
            # cfg = self.train_cfg.copy()
            # prior_grad = [code_.grad.data.clone() for code_ in code_list_]
        # torch.cuda.synchronize()
        # code_time = time.time()
        # print(f'code_time:{code_time-diffuse_time}')
        

        if 'decoder' in optimizer or len(code_optimizers) > 0:
            if len(code_optimizers) > 0:
                code = self.code_activation(torch.stack(code_list_, dim=0))
                # code = [self.code_activation(attr) for attr in self.attr]

            loss_decoder, log_vars_decoder, out_rgbs, target_rgbs = self.loss_decoder_s(
                decoder, code, target_imgs, target_segs, cameras, smpl_params, 
                dt_gamma, cfg=self.train_cfg, densify=(iter > cfg['densify_start_iter']), init=(iter < cfg['init_iter']), cond_norm=cond_norm)
            # for i in range(num_scenes):
            #     img_save = out_rgbs[i]
            #     img_save = img_save.squeeze(0).cpu().detach().numpy()
            #     # img_save = img_save.reshape(5, 4, 128, 128, 3).permute(0, 2, 1, 3, 4).reshape(5*128, 4*128, 3).cpu().detach().numpy()
            #     imageio.imsave(os.path.join('/mnt/sdb/zwt/SSDNeRF/work_dirs/ssdnerf_avatar_uncond_debug/viz_recon_cano','{}.png'.format(data['scene_name'][i])), (255*img_save).astype(np.uint8)) 
            # assert False
            log_vars.update(log_vars_decoder)

            if prior_grad is not None:
                for code_, prior_grad_single in zip(code_list_, prior_grad):
                    code_.grad.copy_(prior_grad_single)
            loss_decoder.backward()

            # if (iter%2) == 1:
            if 'decoder' in optimizer:
                # print(optimizer['decoder'].param_groups[0]['params'][0].grad)
                optimizer['decoder'].step()
            for code_optimizer in code_optimizers:
                code_optimizer.step()
            # for key in optimizer.keys():
            #     optimizer[key].step()

            # ==== save cache ====
            self.save_cache(
                code_list_, code_optimizers, 
                data['scene_id'], data['scene_name'])

            # ==== evaluate reconstruction ====
            with torch.no_grad():
                if len(code_optimizers) > 0:
                    self.mean_ema_update(code)
                train_psnr = eval_psnr(out_rgbs[..., :3], target_rgbs[..., :3])
                code_rms = code.square().flatten(1).mean().sqrt()
                # code_rms = torch.cat(code, dim=1).square().flatten(1).mean().sqrt()
                log_vars.update(train_psnr=float(train_psnr.mean()),
                                code_rms=float(code_rms.mean()))
                if 'test_imgs' in data and data['test_imgs'] is not None:
                    log_vars.update(self.eval_and_viz_s(
                        data, self.decoder, code, smpl_params, cfg=self.train_cfg, ortho=self.ortho, return_norm=self.return_norm)[0])

        # ==== outputs ====
        if 'decoder' in optimizer or len(code_optimizers) > 0:
            log_vars.update(loss_decoder=float(loss_decoder))
        outputs_dict = dict(
            log_vars=log_vars, num_samples=num_scenes)

        # torch.cuda.synchronize()
        # decoder_time = time.time()
        # print(f'decoder_time:{decoder_time-code_time}')
        # assert False
        return outputs_dict

    def val_uncond(self, data, show_pbar=False, **kwargs):
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        num_batches = len(data['scene_id'])
        noise = data.get('noise', None)
        if noise is None:
            noise = torch.randn(
                (num_batches, *self.code_size), device=get_module_device(self))

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            code_out = diffusion(
                self.code_diff_pr(noise), return_loss=False,
                show_pbar=show_pbar, **kwargs)
        code_list = code_out if isinstance(code_out, list) else [code_out]
        # masks_list = []
        # points_list = []
        # density_grid_list = []
        # density_bitfield_list = []
        for step_id, code in enumerate(code_list):
            code = self.code_diff_pr_inv(code)
            n_inverse_steps = self.test_cfg.get('n_inverse_steps', 0)
            if n_inverse_steps > 0 and step_id == (len(code_list) - 1):
                with module_requires_grad(diffusion, False), torch.enable_grad():
                    code_ = self.code_activation.inverse(code).requires_grad_(True)
                    code_optimizer = self.build_optimizer(code_, self.test_cfg)
                    code_scheduler = self.build_scheduler(code_optimizer, self.test_cfg)
                    if show_pbar:
                        pbar = mmcv.ProgressBar(n_inverse_steps)
                    for inverse_step_id in range(n_inverse_steps):
                        code_optimizer.zero_grad()
                        code = self.code_activation(code_)
                        loss, log_vars = diffusion(self.code_diff_pr(code), return_loss=True, cfg=self.test_cfg)
                        loss.backward()
                        code_optimizer.step()
                        if code_scheduler is not None:
                            code_scheduler.step()
                        if show_pbar:
                            pbar.update()
                code = self.code_activation(code_)
            code_list[step_id] = code
            # masks, points = self.get_density(decoder, code, cfg=self.test_cfg)
            # masks_list.append(masks)
            # points_list.append(points)
            # density_grid, density_bitfield = self.get_density(decoder, code, cfg=self.test_cfg)
            # density_grid_list.append(density_grid)
            # density_bitfield_list.append(density_bitfield)
        if isinstance(code_out, list):
            return code_list
            # return code_list, density_grid_list, density_bitfield_list
            # return code_list
        else:
            return code_list[-1]
            # return code_list[-1], density_grid_list[-1], density_bitfield_list[-1]
            # return code_list[-1]

    def val_guide(self, data, **kwargs):
        device = get_module_device(self)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_poses = data['cond_poses']
        smpl_params = data['cond_smpl_param']
        

        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        # (num_scenes, num_imgs, h, w, 3)
        cameras = torch.cat([cond_intrinsics, cond_poses.reshape(num_scenes, num_imgs, -1)], dim=-1)
        dt_gamma = 0.0

        if self.image_cond:
            assert False
            concat_cond = cond_imgs.permute(0, 1, 4, 2, 3)  # (num_scenes, num_imgs, 3, h, w)
            if num_imgs > 1:
                cond_inds = torch.stack([torch.randperm(num_imgs, device=device) for _ in range(num_scenes)], dim=0)
                scene_arange = torch.arange(num_scenes, device=device)[:, None]
                concat_cond = concat_cond[scene_arange, cond_inds]  # (num_scenes, num_imgs, 3, h, w)
            diff_image_size = rgetattr(diffusion, 'denoising.image_size')
            assert diff_image_size[0] % concat_cond.size(-2) == 0
            assert diff_image_size[1] % concat_cond.size(-1) == 0
            concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                            diff_image_size[1] // concat_cond.size(-1)))
        else:
            concat_cond = None

        decoder_training_prev = decoder.training
        decoder.train(True)

        with module_requires_grad(diffusion, False), module_requires_grad(decoder, False):
            def grad_guide_fn(x_0_pred):
                code_pred = self.code_diff_pr_inv(x_0_pred)
                _, loss, _ = self.loss(
                    decoder, code_pred, cond_imgs, cameras, 
                    dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
                    cfg=self.test_cfg, init=False, norm=None)
                return loss * num_scenes

            noise = data.get('noise', None)
            if noise is None:
                noise = torch.randn(
                    (num_scenes, *self.code_size), device=get_module_device(self))

            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                code = diffusion(
                    self.code_diff_pr(noise), return_loss=False,
                    grad_guide_fn=grad_guide_fn, concat_cond=concat_cond, **kwargs)

        decoder.train(decoder_training_prev)
        return self.code_diff_pr_inv(code)

    def val_optim(self, data, code_=None,
                  density_grid=None, density_bitfield=None, show_pbar=False, **kwargs):
        device = get_module_device(self)
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        cond_poses = data['cond_poses']
        smpl_params = data['cond_smpl_param']

        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        # (num_scenes, num_imgs, h, w, 3)
        cameras = torch.cat([cond_intrinsics, cond_poses.reshape(num_scenes, num_imgs, -1)], dim=-1)
        dt_gamma = 0.0

        if self.image_cond:
            assert False
            concat_cond = cond_imgs.permute(0, 1, 4, 2, 3)  # (num_scenes, num_imgs, 3, h, w)
            if num_imgs > 1:
                cond_inds = torch.stack([torch.randperm(num_imgs, device=device) for _ in range(num_scenes)], dim=0)
                scene_arange = torch.arange(num_scenes, device=device)[:, None]
                concat_cond = concat_cond[scene_arange, cond_inds]  # (num_scenes, num_imgs, 3, h, w)
            diff_image_size = rgetattr(diffusion, 'denoising.image_size')
            assert diff_image_size[0] % concat_cond.size(-2) == 0
            assert diff_image_size[1] % concat_cond.size(-1) == 0
            concat_cond = concat_cond.tile((diff_image_size[0] // concat_cond.size(-2),
                                            diff_image_size[1] // concat_cond.size(-1)))
        else:
            concat_cond = None

        decoder_training_prev = decoder.training
        decoder.train(True)

        extra_scene_step = self.test_cfg.get('extra_scene_step', 0)
        # extra_scene_step = 0
        n_inverse_steps = self.test_cfg.get('n_inverse_steps', 100)
        
        assert n_inverse_steps > 0
        if show_pbar:
            pbar = mmcv.ProgressBar(n_inverse_steps)

        with module_requires_grad(diffusion, False), module_requires_grad(decoder, False), torch.enable_grad():
            if code_ is None:
                code_ = self.get_init_code_(num_scenes, cond_imgs.device)
            code_optimizer = self.build_optimizer(code_, self.test_cfg)
            code_scheduler = self.build_scheduler(code_optimizer, self.test_cfg)

            for inverse_step_id in range(n_inverse_steps):
                code_optimizer.zero_grad()
                code = self.code_activation(code_)
                with torch.autocast(
                        device_type='cuda',
                        enabled=self.autocast_dtype is not None,
                        dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                    loss, log_vars = diffusion(
                        self.code_diff_pr(code), return_loss=True,
                        concat_cond=concat_cond[:, inverse_step_id % num_imgs] if concat_cond is not None else None,
                        x_t_detach=self.test_cfg.get('x_t_detach', False),
                        cfg=self.test_cfg, **kwargs)
                loss.backward()

                if extra_scene_step > 0:
                    prior_grad = code_.grad.data.clone()
                    cfg = self.test_cfg.copy()
                    cfg['n_inverse_steps'] = extra_scene_step + 1
                    self.inverse_code(
                        decoder, cond_imgs, cameras, dt_gamma=dt_gamma, smpl_params=smpl_params, cfg=cfg,
                        code_=code_,
                        code_optimizer=code_optimizer,
                        code_scheduler=code_scheduler,
                        prior_grad=prior_grad,
                        densify=False,
                        init=False,
                        cond_norm=None)
                else:  # avoid cloning the grad
                    code = self.code_activation(code_)
                    loss_decoder, log_vars_decoder, out_rgbs, target_rgbs = self.loss_decoder(
                        decoder, code, cond_imgs, cameras, smpl_params, dt_gamma, cfg=self.test_cfg)
                    loss_decoder.backward()
                    code_optimizer.step()
                    if code_scheduler is not None:
                        code_scheduler.step()
                    # print(inverse_step_id)
                if show_pbar:
                    pbar.update()

        decoder.train(decoder_training_prev)
        return self.code_activation(code_)

    def val_step(self, data, viz_dir=None, viz_dir_guide=None, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder
        with torch.no_grad():
            if 'code' in data:
                code = self.load_scene(data, load_density=True)
                # masks, points = self.get_density(decoder, code, cfg=self.test_cfg)
                # print(masks[0].sum())
                # assert False
            elif 'cond_imgs' in data:
                assert False
                cond_mode = self.test_cfg.get('cond_mode', 'guide')
                if cond_mode == 'guide':
                    assert False
                    # code, density_grid, density_bitfield = self.val_guide(data, **kwargs)
                    code = self.val_guide(data, **kwargs)
                elif cond_mode == 'optim':
                    assert False
                    # code, density_grid, density_bitfield = self.val_optim(data, **kwargs)
                    code = self.val_optim(data, **kwargs)
                elif cond_mode == 'guide_optim':
                    code = self.val_guide(data, **kwargs)
                    if viz_dir_guide is not None and 'test_poses' in data:
                        assert False
                    code = self.val_optim(
                        data,
                        code_=self.code_activation.inverse(code).requires_grad_(True),
                        **kwargs)
                else:
                    raise AttributeError
            else:
                # device = get_module_device(self)
                # code = torch.load('/mnt/sdb/zwt/LayerAvatar/cache/ssdnerf_avatar_uncond_16bit_thuman_temp_debug2/viz/scene_0000.pth').to(device)
                # code = code.unsqueeze(0)
                # code = [self.code_activation(attr) for attr in self.attr]
                # code = torch.cat(code, dim=1)
                code = self.val_uncond(data, **kwargs)
            # ==== evaluate reconstruction ====
            if 'test_poses' in data:
                # log_vars, pred_imgs = self.eval_and_viz(
                #     data, decoder, code, density_bitfield,
                #     viz_dir=viz_dir, cfg=self.test_cfg, ortho=self.ortho)
                log_vars, pred_imgs = self.eval_and_viz_s(
                    data, decoder, code,
                    viz_dir=viz_dir, cfg=self.test_cfg, ortho=self.ortho, return_norm=self.return_norm)
            elif 'cond_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz_s(
                    data, decoder, code,
                    viz_dir=viz_dir, cfg=self.test_cfg, ortho=self.ortho, recon=True, return_norm=self.return_norm)
            else:
                log_vars = dict()
                pred_imgs = None
                if viz_dir is None:
                    viz_dir = self.test_cfg.get('viz_dir', None)
                if viz_dir is not None:
                    if isinstance(decoder, DistributedDataParallel):
                        decoder = decoder.module
                    decoder.visualize(
                        code, data['scene_name'],
                        viz_dir, code_range=self.test_cfg.get('clip_range', [-1, 1]))
        # print(time.time()-s_time)
        # assert False
        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            # self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])
            self.save_scene(save_dir, code, data['scene_name'])
            save_mesh = self.test_cfg.get('save_mesh', False)
            if save_mesh:
                mesh_resolution = self.test_cfg.get('mesh_resolution', 256)
                mesh_threshold = self.test_cfg.get('mesh_threshold', 10)
                self.save_mesh(save_dir, decoder, code, data['scene_name'], mesh_resolution, mesh_threshold)
        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)
        return outputs_dict