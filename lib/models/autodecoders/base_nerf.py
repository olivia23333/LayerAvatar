import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import lpips
import mmcv
import trimesh
import cv2
import json

from copy import deepcopy
from glob import glob
import time
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.nn.functional as F
# from typing import Iterable, List, Tuple, Union
# from torch import Tensor
# from pytorch3d.renderer import FoVPerspectiveCameras
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
from mmcv.runner import load_checkpoint
from mmgen.models.builder import MODULES, build_module
from mmgen.models.architectures.common import get_module_device
from pytorch3d.ops import knn_points

from ...core import custom_meshgrid, eval_psnr, eval_ssim_skimage, reduce_mean, rgetattr, rsetattr, extract_geometry, \
    module_requires_grad, get_cam_rays
from lib.ops import morton3D, morton3D_invert, packbits
from lib.meshudf.meshudf import get_mesh_from_udf
import open3d as o3d
from lib.models.renderers import batch_rodrigues

LPIPS_BS = 32


@MODULES.register_module()
class TanhCode(nn.Module):
    def __init__(self, scale=1.0, eps=1e-5):
        super(TanhCode, self).__init__()
        self.scale = scale
        self.eps = eps

    def forward(self, code_, update_stats=False):
        return code_.tanh() if self.scale == 1 else code_.tanh() * self.scale

    def inverse(self, code):
        return code.clamp(min=-1 + self.eps, max=1 - self.eps).atanh() if self.scale == 1 \
            else (code / self.scale).clamp(min=-1 + self.eps, max=1 - self.eps).atanh()

@MODULES.register_module()
class IdentityCode(nn.Module):
    @staticmethod
    def forward(code_, update_stats=False):
        return code_

    @staticmethod
    def inverse(code):
        return code


@MODULES.register_module()
class NormalizedTanhCode(nn.Module):
    def __init__(self, mean=0.0, std=1.0, clip_range=1, eps=1e-5, momentum=0.001):
        super(NormalizedTanhCode, self).__init__()
        self.mean = mean
        self.std = std
        self.clip_range = clip_range
        self.register_buffer('running_mean', torch.tensor([0.0]))
        self.register_buffer('running_var', torch.tensor([std ** 2]))
        self.momentum = momentum
        self.eps = eps

    def forward(self, code_, update_stats=False):
        if update_stats and self.training:
            with torch.no_grad():
                var, mean = torch.var_mean(code_)
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(mean))
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * reduce_mean(var))
        scale = (self.std / (self.running_var.sqrt() + self.eps)).to(code_.device)
        return (code_ * scale + (self.mean - self.running_mean.to(code_.device) * scale)
                ).div(self.clip_range).tanh().mul(self.clip_range)

    def inverse(self, code):
        scale = ((self.running_var.sqrt() + self.eps) / self.std).to(code.device)
        return code.div(self.clip_range).clamp(min=-1 + self.eps, max=1 - self.eps).atanh().mul(
            self.clip_range * scale) + (self.running_mean.to(code.device) - self.mean * scale)

def huber(x, y, scaling=0.1):
    """
    A helper function for evaluating the smooth L1 (huber) loss
    between the rendered silhouettes and colors.
    """
    # import ipdb; ipdb.set_trace()
    diff_sq = (x - y) ** 2
    loss = ((1 + diff_sq / (scaling**2)).clamp(1e-4).sqrt() - 1) * float(scaling)
    # if mask is not None:
    #     loss = loss.abs().sum()/mask.sum()
    # else:
    loss = loss.abs().mean()
    return loss

# def get_o3d_mesh_from_tensors(
#     vertices: Union[Tensor, np.ndarray],
#     triangles: Union[Tensor, np.ndarray],
# ) -> o3d.geometry.TriangleMesh:
#     """Get open3d mesh from either numpy arrays or torch tensors.
#     The input vertices must have shape (NUM_VERTICES, D), where D
#     can be 3 (only X,Y,Z), 6 (X,Y,Z and normals) or 9 (X,Y,Z, normals and colors).
#     The input triangles must have shape (NUM_TRIANGLES, D), where D can be 3
#     (only vertex indices) or 6 (vertex indices and normals).
#     Args:
#         vertices: The numpy array or torch tensor with vertices
#             with shape (NUM_VERTICES, D).
#         triangles: The numpy array or torch tensor with triangles
#             with shape (NUM_TRIANGLES, D).
#     Returns:
#         The open3d mesh.
#     """
#     mesh_o3d = o3d.geometry.TriangleMesh()

#     if isinstance(vertices, Tensor):
#         v = vertices.clone().detach().cpu().numpy()
#     else:
#         v = np.copy(vertices)

#     if isinstance(triangles, Tensor):
#         t = triangles.clone().detach().cpu().numpy()
#     else:
#         t = np.copy(triangles)

#     mesh_o3d.vertices = o3d.utility.Vector3dVector(v[:, :3])

#     if v.shape[1] == 6:
#         mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(v[:, 3:6])

#     if v.shape[1] == 9:
#         mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(v[:, 6:9])

#     mesh_o3d.triangles = o3d.utility.Vector3iVector(t[:, :3])

#     if t.shape[1] == 6:
#         mesh_o3d.triangle_normals = o3d.utility.Vector3dVector(t[:, 3:6])

#     return mesh_o3d

class BaseNeRF(nn.Module):
    def __init__(self,
                 code_size=(3, 8, 64, 64),
                 code_activation=dict(
                     type='TanhCode',
                     scale=1),
                 grid_size=64,
                 decoder=dict(
                     type='TriPlaneDecoder'),
                 decoder_use_ema=False,
                 bg_color=1,
                 pixel_loss=dict(
                     type='MSELoss'),
                 scale_loss_weight=0,
                 pixel_loss_weight=0,
                 seg_loss_weight=0,
                 init_seg_weight=0,
                 inner_loss_weight=0,
                 skin_loss_weight=0,
                 opacity_loss_weight=0,
                #  norm_reg_weight=0,
                #  alpha_reg_weight=0,
                 reg_loss=None,
                 reg_code_loss=None,
                 per_loss=None,
                 update_extra_interval=16,
                 use_lpips_metric=True,
                 init_from_mean=False,
                 init_scale=1e-4,
                 mean_ema_momentum=0.001,
                 mean_scale=1.0,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 pretrained=None):
        super().__init__()
        self.code_size = code_size
        self.code_activation = build_module(code_activation)
        self.grid_size = grid_size
        self.decoder = build_module(decoder)
        self.decoder_use_ema = decoder_use_ema
        if self.decoder_use_ema:
            self.decoder_ema = deepcopy(self.decoder)
        self.bg_color = bg_color
        # self.pixel_loss = build_module(pixel_loss)
        self.reg_loss = build_module(reg_loss) if reg_loss is not None else None
        self.reg_code_loss = build_module(reg_code_loss) if reg_code_loss is not None else None
        self.per_loss = build_module(per_loss) if per_loss is not None else None
        self.scale_loss_weight = scale_loss_weight
        self.pixel_loss_weight = pixel_loss_weight
        self.seg_loss_weight = seg_loss_weight
        self.init_seg_weight = init_seg_weight
        self.inner_loss_weight = inner_loss_weight
        self.skin_loss_weight = skin_loss_weight
        self.opacity_loss_weight = opacity_loss_weight
        # self.norm_reg_weight = norm_reg_weight
        # self.alpha_reg_weight = alpha_reg_weight
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.update_extra_interval = update_extra_interval
        self.lpips = [] if use_lpips_metric else None  # use a list to avoid registering the LPIPS model in state_dict
        if init_from_mean:
            self.register_buffer('init_code', torch.zeros(code_size))
        # elif init_from_pre:
        #     self.register_buffer('init_code', torch.zeros(code_size))
        else:
            self.init_code = None
        self.init_scale = init_scale
        self.mean_ema_momentum = mean_ema_momentum
        self.mean_scale = mean_scale
        if pretrained is not None and os.path.isfile(pretrained):
            load_checkpoint(self, pretrained, map_location='cpu')

        self.train_cfg_backup = dict()
        for key, value in self.test_cfg.get('override_cfg', dict()).items():
            self.train_cfg_backup[key] = rgetattr(self, key, None)

    def train(self, mode=True):
        if mode:
            for key, value in self.train_cfg_backup.items():
                rsetattr(self, key, value)
        else:
            for key, value in self.test_cfg.get('override_cfg', dict()).items():
                if self.training:
                    self.train_cfg_backup[key] = rgetattr(self, key)
                rsetattr(self, key, value)
        super().train(mode)
        return self

    def gmof(self, pred, gt, sigma=10):
        """
        Geman-McClure error function
        """
        x = gt - pred
        x_squared = x ** 2
        sigma_squared = sigma ** 2
        error = (sigma_squared * x_squared) / (sigma_squared + x_squared)
        return error.mean()

    def load_scene(self, data, load_density=False):
        device = get_module_device(self)
        code_list = []
        # masks = []
        # points = []
        # density_grid = []
        # density_bitfield = []
        for code_state_single in data['code']:
            code_list.append(
                code_state_single['param']['code'] if 'code' in code_state_single['param']
                else self.code_activation(code_state_single['param']['code_']))
            # if load_density:
                # density_grid.append(code_state_single['param']['density_grid'])
                # density_bitfield.append(code_state_single['param']['density_bitfield'])
            # if load_density:
                # if 'masks' in code_state_single['param']:
                # masks.append(code_state_single['param']['masks'].bool())
                # points.append(code_state_single['param']['points'])
        code = torch.stack(code_list, dim=0).to(device)
        # masks = torch.stack(masks, dim=0).to(device) if load_density else None
        # points = torch.stack(points, dim=0).to(device) if load_density else None
        # masks = None
        # points = None
        # density_grid = torch.stack(density_grid, dim=0).to(device) if load_density else None
        # density_bitfield = torch.stack(density_bitfield, dim=0).to(device) if load_density else None
        # return code, density_grid, density_bitfield
        return code

    @staticmethod
    def save_scene(save_dir, code, scene_name):
    # def save_scene(save_dir, code, density_grid, density_bitfield, scene_name):
        os.makedirs(save_dir, exist_ok=True)
        for scene_id, scene_name_single in enumerate(scene_name):
            results = dict(
                scene_name=scene_name_single,
                param=dict(
                    code=code.data[scene_id].cpu(),
                    # density_grid=density_grid.data[scene_id].cpu(),
                    # density_bitfield=density_bitfield.data[scene_id].cpu()
                    ))
            torch.save(results, os.path.join(save_dir, scene_name_single) + '.pth')

    @staticmethod
    def save_mesh(save_dir, decoder, code, scene_name, mesh_resolution, mesh_threshold):
        os.makedirs(save_dir, exist_ok=True)
        for code_single, scene_name_single in zip(code, scene_name):
            vertices, triangles = extract_geometry(
                decoder,
                code_single,
                mesh_resolution,
                mesh_threshold)
            mesh = trimesh.Trimesh(vertices, triangles, process=False)
            mesh.export(os.path.join(save_dir, scene_name_single) + '.stl')

    def get_init_code_(self, num_scenes, device=None):
        code_ = torch.empty(
            self.code_size if num_scenes is None else (num_scenes, *self.code_size),
            device=device, requires_grad=True, dtype=torch.float32)
        if self.init_code is None:
            code_.data.uniform_(-self.init_scale, self.init_scale)
        else:
            code_.data[:] = self.code_activation.inverse(self.init_code * self.mean_scale)
        return code_
    
    def get_init_mask_(self, num_scenes, num_init, device=None):
        mask = torch.ones(num_init if num_scenes is None else (num_scenes, num_init), device=device, dtype=torch.bool)
        return mask

    def get_init_points_(self, num_scenes, init_pcd, device=None):
        if num_scenes == None:
            points = init_pcd
        else:
            points = init_pcd[None].expand(num_scenes, -1, -1)
        # points = torch.zeros((num_init, 3) if num_scenes is None else (num_scenes, num_init, 3), device=device, dtype=torch.float16)
        return points

    def get_init_density_grid(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 if num_scenes is None else (num_scenes, self.grid_size ** 3),
            device=device, dtype=torch.float16)

    def get_init_density_bitfield(self, num_scenes, device=None):
        return torch.zeros(
            self.grid_size ** 3 // 8 if num_scenes is None else (num_scenes, self.grid_size ** 3 // 8),
            device=device, dtype=torch.uint8)

    @staticmethod
    def build_optimizer(code_, cfg):
        # optimizer_cfg = cfg['optimizer'].copy()
        optimizer_cfg = deepcopy(cfg['optimizer'])
        # optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        # if isinstance(code_, list):
        #     optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
        #     code_optimizer = [
        #         optimizer_class([code_single_], **optimizer_cfg)
        #         for code_single_ in code_]
        if isinstance(optimizer_cfg, list):
            code_attr = torch.split(code_[0], [3, 9, 9, 9, 9], dim=0)
            code_attr = [nn.Parameter(code_a, requires_grad=True) for code_a in code_attr]
            code_optimizer = []
            for idx in range(len(code_attr)):
                optimizer_class = getattr(torch.optim, optimizer_cfg[idx].pop('type'))
                code_optimizer.append(optimizer_class([code_attr[idx]], **optimizer_cfg[idx]))
            # code_[0] = torch.cat(code_attr, dim=0)
            code_[0] = code_attr
        elif isinstance(code_, list):
            optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
            code_optimizer = [
                optimizer_class([code_single_], **optimizer_cfg)
                for code_single_ in code_]
        else:
            optimizer_class = getattr(torch.optim, optimizer_cfg.pop('type'))
            code_optimizer = optimizer_class([code_], **optimizer_cfg)
        return code_optimizer

    @staticmethod
    def build_scheduler(code_optimizer, cfg):
        if 'lr_scheduler' in cfg:
            scheduler_cfg = cfg['lr_scheduler'].copy()
            scheduler_class = getattr(torch.optim.lr_scheduler, scheduler_cfg.pop('type'))
            if isinstance(code_optimizer, list):
                code_scheduler = [
                    scheduler_class(code_optimizer_single, **scheduler_cfg)
                    for code_optimizer_single in code_optimizer]
            else:
                code_scheduler = scheduler_class(code_optimizer, **scheduler_cfg)
        else:
            code_scheduler = None
        return code_scheduler

    @staticmethod
    def ray_sample(cond_rays_o, cond_rays_d, cond_imgs, n_samples, sample_inds=None, ortho=True):
        """
        Args:
            cond_rays_o (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            cond_rays_d (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            cond_imgs (torch.Tensor): (num_scenes, num_imgs, h, w, 3)
            n_samples (int): number of samples
            sample_inds (None | torch.Tensor): (num_scenes, n_samples)

        Returns:
            rays_o (torch.Tensor): (num_scenes, n_samples, 3)
            rays_d (torch.Tensor): (num_scenes, n_samples, 3)
            target_rgbs (torch.Tensor): (num_scenes, n_samples, 3)
        """
        device = cond_rays_o.device
        num_scenes, num_imgs, h, w, _ = cond_rays_o.size()
        num_scene_pixels = num_imgs * h * w
        rays_o = cond_rays_o.reshape(num_scenes, num_scene_pixels, 3)
        rays_d = cond_rays_d.reshape(num_scenes, num_scene_pixels, 3)
        # smpl_param = smpl_params.reshape(num_scenes, num_scene_pixels, )
        target_rgbs = cond_imgs.reshape(num_scenes, num_scene_pixels, 3)
        
        if num_scene_pixels > n_samples:
            if sample_inds is None:
                sample_inds = [torch.randperm(
                    target_rgbs.size(1), device=device)[:n_samples] for _ in range(num_scenes)]
                sample_inds = torch.stack(sample_inds, dim=0)
            scene_arange = torch.arange(num_scenes, device=device)[:, None]
            rays_o = rays_o[scene_arange, sample_inds]
            rays_d = rays_d[scene_arange, sample_inds]
            target_rgbs = target_rgbs[scene_arange, sample_inds]
        return rays_o, rays_d, target_rgbs

    @staticmethod
    def get_raybatch_inds(cond_imgs, n_inverse_rays):
        device = cond_imgs.device
        num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        num_scene_pixels = num_imgs * h * w
        if num_scene_pixels > n_inverse_rays:
            raybatch_inds = [torch.randperm(num_scene_pixels, device=device) for _ in range(num_scenes)]
            raybatch_inds = torch.stack(raybatch_inds, dim=0).split(n_inverse_rays, dim=1)
            num_raybatch = len(raybatch_inds)
        else:
            raybatch_inds = num_raybatch = None
        return raybatch_inds, num_raybatch

    # def loss(self, decoder, code, density_bitfield, target_rgbs, cameras,
    #         dt_gamma=0.0, smpl_params=None, return_decoder_loss=False, scale_num_ray=1.0,
    #          cfg=dict(), **kwargs):
    def loss(self, decoder, code, target_rgbs, target_segs, cameras,   
            dt_gamma=0.0, smpl_params=None, return_decoder_loss=False, scale_num_ray=1.0,
             cfg=dict(), init=False, norm=None, **kwargs):
        # target_rgbs min:0, max:1 shape: num_scenes, num_imgs, h, w, 3
        num_batch, num_imgs, h, w = target_rgbs.shape[:4]
        outputs = decoder(
            code, self.grid_size, smpl_params, cameras,
            num_imgs, dt_gamma=dt_gamma, perturb=True, return_loss=return_decoder_loss, init=init, return_norm=(norm!=None))

        # out_rgbs, out_part = torch.split(outputs['image'], [3, 3, 3, 3], dim=-1)
        out_rgbs, out_part = outputs['image'][..., :3], outputs['image'][..., 3:]
        # print(out_rgbs.shape)
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_rgbs_debug.png', (out_rgbs[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_rgbs_debug.png', (target_rgbs[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        out_seg_inner, out_segs, out_seg_part = outputs['segs'][..., :1], outputs['segs'][..., 1:4], outputs['segs'][..., 4:]
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_0_1.png', (torch.cat([out_part[0, 0, :, :, :3], out_seg_part[0, 0, :, :, :1]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_1_1.png', (torch.cat([out_part[0, 0, :, :, 3:6], out_seg_part[0, 0, :, :, 1:2]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_2_1.png', (torch.cat([out_part[0, 0, :, :, 6:9], out_seg_part[0, 0, :, :, 2:3]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_3_1.png', (torch.cat([out_part[0, 0, :, :, 9:12], out_seg_part[0, 0, :, :, 3:4]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_4_1.png', (torch.cat([out_part[0, 0, :, :, 12:], out_seg_part[0, 0, :, :, 4:]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        # out_segs, out_seg_part = outputs['segs'][..., :1], outputs['segs'][..., 1:]
        # out_depths, out_depth_part = outputs['depths'][..., :1], outputs['depths'][..., 1:]
        # out_seg, seg_cloth, seg_body, seg_hair = torch.split(outputs['segs'], [3, 1, 1, 1], dim=-1)

        # out_seg, seg_cloth, seg_body, seg_hair = torch.split(outputs['segs'], [3, 3, 3, 3], dim=-1)
        # for 3d gaussian
        # seg_cloth = out_segs[..., 3:6]
        # seg_body = out_segs[..., 6:9]
        # seg_hair = out_segs[..., 9:]
        out_offsets = outputs['offset']
        attrmap = outputs['attrmap']
        # scales = outputs['scales']
        # reg_collision = outputs['collision']
        # out_edge = outputs['edge']
        skin_color = outputs['reg_color'].detach()
        # skin_color = outputs['skin_color']
        # reg_dists = outputs['reg_dists']
        # reg_norms = outputs['reg_norms']
        # depth gaussian
        target_seg_full = target_segs[..., 5:].expand(-1, -1, -1, -1, 3)
        target_seg_idx = target_segs[..., 5:].clone().long()
        target_seg_part = target_segs[..., :5].float()
        non_skin_mask = target_segs[..., 1:5].sum(-1).bool().float()
        # skin_mask = target_segs[..., :1].float() * out_seg_part[..., 0].detach()
        # skin_color = (out_part[..., :3][skin_mask])
        fg_mask = target_segs[..., :5].sum(-1, keepdim=True).bool().float()
        occulusion_mask = non_skin_mask * out_seg_part[..., 0].detach()
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_segs.png', (out_segs[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        # for 3d gaussian
        # target_seg_full = target_segs[..., 3:].expand(-1, -1, -1, -1, 3)
        # target_seg_body = target_segs[..., [0]].expand(-1, -1, -1, -1, 3)
        # target_seg_cloth = target_segs[..., [1]].expand(-1, -1, -1, -1, 3)
        # target_seg_hair = target_segs[..., [2]].expand(-1, -1, -1, -1, 3)
        # non_skin_mask = (target_segs[..., [1]].bool() + target_segs[..., [2]].bool()).expand(-1, -1, -1, -1, 3).float()
        # for 2d gaussian
        # target_seg_full = target_segs[..., 3:].expand(-1, -1, -1, -1, 3)
        # target_seg_body = target_segs[..., [0]]
        # target_seg_cloth = target_segs[..., [1]]
        # target_seg_hair = target_segs[..., [2]]
        # non_skin_mask = (target_segs[..., [1]].bool() + target_segs[..., [2]].bool()).float()
        
        # target_seg_full = target_segs[..., 3:].expand(-1, -1, -1, -1, 3)
        # target_seg_body = target_segs[..., [0]]
        # target_seg_cloth = target_segs[..., [1]]
        # target_seg_hair = target_segs[..., [2]]
        # non_skin_mask = (target_segs[..., [1]].bool() + target_segs[..., [2]].bool()).float()
        # out_opacity = outputs['sigmas']
        # full_mask = (target_segs.bool().sum(-1)).unsqueeze(-1).expand(-1, -1, -1, -1, 3)
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/gt_rgb.jpg', ((target_rgbs)[2, 2].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/pred_rgb.jpg', ((out_rgbs)[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/pred_cloth_rgb.jpg', ((out_cloth)[2, 2].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/pred_hair_rgb.jpg', ((out_hair)[2, 2].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        # init = True
        loss = 0
        scale = 1 - math.exp(-cfg['loss_coef'] * scale_num_ray) if 'loss_coef' in cfg else 1
        seg_part = target_seg_part.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3).flatten(4, 5)
        # if init:
        #     target_seg_idx = target_seg_idx - 1
        #     valid_mask =(~(target_seg_idx < 0))
        #     target_seg_idx[~valid_mask] = 0
        #     predict_rgb = torch.gather(out_part.reshape(num_batch, num_imgs, h, w, 5, 3), dim=-2, index=target_seg_idx.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3))[:, :, :, :, 0]
        #     predict_seg = torch.gather(out_seg_part.reshape(num_batch, num_imgs, h, w, 5, 1), dim=-2, index=target_seg_idx.unsqueeze(-1))[:, :, :, :, 0]
        #     seg_pixel_loss = huber(predict_seg, fg_mask) * (scale) # 2.5
        #     # pixel_loss_full = huber(predict_rgb[valid_mask[..., 0]], target_rgbs[valid_mask[..., 0]]) * (scale * 3) # 2.5
        #     pixel_loss_full = huber(predict_rgb, target_rgbs) * (scale * 3) # 2.5
        # else:
        pixel_loss_full = huber(out_rgbs, target_rgbs) * (scale * 3) # 2.5
        pixel_loss_part = huber(out_part*seg_part, target_rgbs.repeat(1, 1, 1, 1, 5)*seg_part) * (scale*3)
        pixel_loss = (pixel_loss_full*0.5 + pixel_loss_part*0.5) * self.pixel_loss_weight
        # print(out_part.shape)
        # print(seg_part.shape)
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_0_1_tar.png', (torch.cat([out_part[0, 0, :, :, :3], seg_part[0, 0, :, :, :1]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_1_1_tar.png', (torch.cat([out_part[0, 0, :, :, 3:6], seg_part[0, 0, :, :, 3:4]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_2_1_tar.png', (torch.cat([out_part[0, 0, :, :, 6:9], seg_part[0, 0, :, :, 6:7]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_3_1_tar.png', (torch.cat([out_part[0, 0, :, :, 9:12], seg_part[0, 0, :, :, 9:10]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part_4_1_tar.png', (torch.cat([out_part[0, 0, :, :, 12:], seg_part[0, 0, :, :, 12:13]], dim=-1).detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        # pixel_loss = pixel_loss_full
        # if norm != None:
        #     out_norms = outputs['norm']
        #     pix_norm_loss = self.pixel_loss(out_norms, norm, **kwargs) * (scale * 3) # 2.5
        #     pixel_loss = pixel_loss * 0.7 + pix_norm_loss * 0.3
        loss = loss + pixel_loss

        loss_dict = dict(pixel_loss=pixel_loss)
        # occulusion_mask_cloth = 1 - (target_segs[..., 0] * out_segs[..., :3].max(-1)[0].detach())
        # occulusion_mask_hair = 1 - (target_segs[..., 0] * out_segs[..., 3:].max(-1)[0].detach())
        # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/seg_loss_pred.png', (((out_segs[..., :3].max(-1)[0])*occulusion_mask_cloth)[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/seg_loss_gt.png', ((target_segs[..., 1]*occulusion_mask_cloth)[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        # skin_pred_mask = target_segs[..., 0].bool()
        # occulusion_mask = (~skin_pred_mask).float()
        # print(out_seg.shape)
        # print(target_seg_full.shape)
        # assert False
        # predict_seg = torch.gather(out_seg_part, dim=-1, index=target_seg_idx)
        # fg_loss_full = self.pixel_loss(predict_seg, fg_mask, **kwargs) * scale * 0.5 # 2.5
        # seg_loss_full = huber(out_segs, (target_seg_full*0.2)) * scale # 2.5
        seg_loss_part = huber(out_seg_part[..., 1:], target_seg_part[..., 1:]) * scale
        seg_loss_body = huber(out_seg_part[..., :1]*target_seg_part[..., :1], target_seg_part[..., :1]) * scale
        pred_non_skin = 1 - out_seg_part[..., :1].detach()
        pred_skin_mask = (non_skin_mask.unsqueeze(-1) * pred_non_skin) + out_seg_part[..., :1]
        seg_loss_skin = huber(pred_skin_mask, fg_mask) * scale
        seg_loss = (seg_loss_part*0.4 + seg_loss_body*0.05 + seg_loss_skin*0.05) * self.seg_loss_weight
        # seg_loss = (seg_loss_full*0.5 + seg_loss_part*0.4 + seg_loss_body*0.05 + seg_loss_skin*0.05) * self.seg_loss_weight
        # if init:
        #     seg_loss += seg_pixel_loss * self.init_seg_weight
        # seg_loss = seg_loss_part
        # seg_loss = seg_loss_full
        loss = loss + seg_loss 
        loss_dict.update(seg_loss=seg_loss)

        # depth order loss
        
        # predict_depth = torch.gather(out_depth_part, dim=-1, index=target_seg_idx)
        # valid_mask = torch.logical_and(valid_mask, out_segs > 0.5)[..., 0]
        # # valid_mask = torch.logical_and(valid_mask, predict_depth < 999)[..., 0]
        # out_depths_ = out_depths[valid_mask]
        # predict_depth_ = predict_depth[valid_mask]
        # exclude_mask = ~(predict_depth_ == out_depths_)
        # if exclude_mask.sum() != 0:
        #     depth_order_loss = torch.log(1 + torch.exp(predict_depth_[exclude_mask] - out_depths_[exclude_mask])).mean()
        #     depth_order_loss *= 1e-2
        # else:
        #     depth_order_loss = torch.zeros_like(seg_loss)

        # loss = loss + depth_order_loss
        # loss_dict.update(depth_order_loss=depth_order_loss)

        # collision loss
        # normal loss

        # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/seg_body_pred.png', ((out_seg[..., 6:].max(-1)[0])[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/seg_body_gt.png', ((target_segs[..., 3])[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        seg_inner_loss = F.relu(out_seg_inner - fg_mask).mean() * self.inner_loss_weight
        loss = loss + seg_inner_loss
        loss_dict.update(seg_inner_loss=seg_inner_loss)

        # seg_body_loss = self.pixel_loss((out_segs[..., 6:].max(-1)[0]) + (1 - out_segs[..., 6:].max(-1)[0]).detach() * non_skin_mask[..., 0].float(), full_mask[..., 0].float(), **kwargs) * scale * 0.01
        # loss = loss + seg_body_loss 
        # loss_dict.update(seg_body_loss=seg_body_loss)
        
        reg_offset = out_offsets * cfg['offset_weight'] 
        loss = loss + reg_offset
        loss_dict.update(reg_offset=reg_offset)

        # reg_scale = F.relu(5 - scales[:, :, :2] / scales[:, :, 2:]).sum() * self.scale_loss_weight
        # if reg_scale > 1e-4:
        #     loss = loss + reg_scale
        #     loss_dict.update(reg_scale=reg_scale)
        # else:
        #     loss_dict.update(reg_scale=0.)

        # reg_edge = out_edge * 500
        # loss = loss + reg_edge
        # loss_dict.update(reg_edge=reg_edge)

        # reg_code = (code**2).mean()
        # loss = loss + reg_code * 1e-2
        # loss_dict.update(reg_code=reg_code*1e-2)

        # reg_dists = reg_dists * 1000
        # loss = loss + reg_dists
        # loss_dict.update(reg_dists=reg_dists)

        # reg_norms = reg_norms * 0.005
        # loss = loss + reg_norms
        # loss_dict.update(reg_norms=reg_norms)
        
        # skin color consistency
        # print(out_part.shape)
        # print(occulusion_mask.shape)
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_part.png', ((out_part[0, 0, :, :, :3]*occulusion_mask[0, 0].unsqueeze(-1)).detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/occulusion_mask.png', (occulusion_mask[0, 0].cpu().numpy()*255).astype(np.uint8))
        # assert False

        reg_rgb = ((out_part[..., :3] - skin_color.reshape(-1, 1, 1, 1, 3))*occulusion_mask.unsqueeze(-1)).abs().mean() * self.skin_loss_weight
        loss = loss + reg_rgb 
        loss_dict.update(reg_rgb=reg_rgb)

        # loss = loss + reg_color
        # loss_dict.update(reg_color=reg_color)
        # loss = loss + reg_collision 
        # loss_dict.update(reg_collision=reg_collision)

        val = torch.clamp(out_seg_part[..., :1], 1e-3, 1 - 1e-3)
        reg_opacity = (val*torch.log(val) + (1-val)*torch.log(1-val)).mean() * -1 * self.opacity_loss_weight
        loss = loss + reg_opacity
        loss_dict.update(reg_opacity=reg_opacity)

        # reg_scale = torch.sum(torch.prod(out_scales, dim=-1)) * cfg['scale_weight']
        # loss = loss + reg_scale
        # loss_dict.update(reg_scale=reg_scale)
        if self.reg_loss is not None:
            reg_loss = self.reg_loss(attrmap, **kwargs)
            loss = loss + reg_loss
            loss_dict.update(reg_loss=reg_loss)
        if self.reg_code_loss is not None:
            reg_code_loss = self.reg_code_loss(code, **kwargs)
            loss = loss + reg_code_loss
            loss_dict.update(reg_code_loss=reg_code_loss)
        if self.per_loss is not None:
            if self.per_loss.loss_weight > 0 and (not init):
                per_loss = self.per_loss(out_rgbs[:, 0], target_rgbs[:, 0])
                # if norm != None:
                #     per_loss_norm = self.per_loss(out_norms[:, 0], target_rgbs[:, 0])
                #     per_loss = per_loss * 0.7 + per_loss_norm * 0.3
                loss = loss + per_loss
                loss_dict.update(per_loss=per_loss)
        # if self.scale_loss_weight > 0:
        #     out_scales = outputs['scales'][..., :2]
        #     scale_reg_loss = (out_scales.max(-1).values/out_scales.min(-1).values - 5).clip(min=0).mean() * self.scale_loss_weight
        #     loss = loss + scale_reg_loss
        #     loss_dict.update(scale_reg_loss=scale_reg_loss)
        # if self.norm_reg_weight > 0:
        #     out_alphas = outputs['alphas'][..., 0]
        #     norm_reg_loss = (out_alphas.detach() * (1.0 - torch.sum(outputs['norm'] * outputs['norm_maps'], axis=-1))).mean() * self.norm_reg_weight
        #     loss = loss + norm_reg_loss
        #     loss_dict.update(norm_reg_loss=norm_reg_loss)
        # if self.alpha_reg_weight > 0:
        #     val = torch.clamp(out_alphas, 1e-3, 1 - 1e-3)
        #     alpha_reg_loss = torch.mean(torch.log(val) + torch.log(1 - val)) * self.alpha_reg_weight
        #     loss = loss + alpha_reg_loss
        #     loss_dict.update(alpha_reg_loss=alpha_reg_loss)
        if return_decoder_loss and outputs['decoder_reg_loss'] is not None:
            decoder_reg_loss = outputs['decoder_reg_loss']
            loss = loss + decoder_reg_loss
            loss_dict.update(decoder_reg_loss=decoder_reg_loss)
        return out_rgbs, loss, loss_dict
    
    def loss_t(self, decoder, code, target_rgbs, target_segs, cameras,   
            dt_gamma=0.0, smpl_params=None, return_decoder_loss=False, scale_num_ray=1.0,
             cfg=dict(), init=False, norm=None, **kwargs):
        # target_rgbs min:0, max:1 shape: num_scenes, num_imgs, h, w, 3
        num_batch, num_imgs, h, w = target_rgbs.shape[:4]
        outputs = decoder(
            code, self.grid_size, smpl_params, cameras,
            num_imgs, dt_gamma=dt_gamma, perturb=True, return_loss=return_decoder_loss, init=init, return_norm=(norm!=None))

        out_rgbs, out_part = outputs['image'][..., :3], outputs['image'][..., 3:]
        out_seg_inner, out_segs, out_seg_part = outputs['segs'][..., :1], outputs['segs'][..., 1:2], outputs['segs'][..., 2:]
        # print(outputs['image'].shape)
        # print(outputs['segs'].shape)
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l3.png', (out_seg_part[0, 0, :, :, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l3.png', (target_segs[0, 0, :, :, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_l0.png', (out_rgbs[0, 0, :, :, :3].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_l0.png', (target_rgbs[0, 0, :, :, :3].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l0.png', (out_seg_part[0, 0, :, :, 1].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l0.png', (target_segs[0, 0, :, :, 1].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l1.png', (out_seg_part[0, 0, :, :, 2].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l1.png', (target_segs[0, 0, :, :, 2].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l2.png', (out_seg_part[0, 0, :, :, 3].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l2.png', (target_segs[0, 0, :, :, 3].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        # out_rgbs = outputs['image']
        # out_seg_inner, out_seg_part = outputs['segs'][..., :1], outputs['segs'][..., 1:2], outputs['segs'][..., 2:]
        
        out_offsets = outputs['offset']
        attrmap = outputs['attrmap']
        scales = outputs['scales']
        skin_color = outputs['reg_color'].detach()

        target_seg_full = target_segs[..., :1].float()
        target_seg_part = target_segs[..., 1:].float()
        non_skin_mask = target_segs[..., 1:].sum(-1).bool().float()
        # fg_mask = target_segs[..., :1].bool().float()
        occulusion_mask = non_skin_mask * out_seg_part[..., 0].detach()
        
        loss = 0
        scale = 1 - math.exp(-cfg['loss_coef'] * scale_num_ray) if 'loss_coef' in cfg else 1
        # seg_part = target_seg_part.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3).flatten(4, 5)
        pixel_loss_full = huber(out_rgbs, target_rgbs[..., :3]) * (scale * 3) # 2.5
        pixel_loss_part = huber(out_part[..., 3:], target_rgbs[..., 3:]) * (scale * 3) # 2.5
        pixel_loss = (pixel_loss_full*0.5 + pixel_loss_part*0.5) * self.pixel_loss_weight
        loss = loss + pixel_loss
        loss_dict = dict(pixel_loss=pixel_loss)
        
        seg_loss_full = huber(out_segs, target_seg_full) * scale
        seg_loss_part = huber(out_seg_part[..., 1:], target_seg_part) * scale
        pred_non_skin = 1 - out_seg_part[..., :1].detach()
        pred_skin_mask = (non_skin_mask.unsqueeze(-1) * pred_non_skin) + out_seg_part[..., :1]
        seg_loss_skin = huber(pred_skin_mask, target_seg_full) * scale
        seg_loss = (seg_loss_full*0.5 + seg_loss_part*0.4 + seg_loss_skin*0.1) * self.seg_loss_weight
        loss = loss + seg_loss 
        loss_dict.update(seg_loss=seg_loss)

        seg_inner_loss = F.relu(out_seg_inner - target_seg_full).mean() * self.inner_loss_weight
        loss = loss + seg_inner_loss
        loss_dict.update(seg_inner_loss=seg_inner_loss)
        
        reg_offset = out_offsets * cfg['offset_weight'] 
        loss = loss + reg_offset
        loss_dict.update(reg_offset=reg_offset)

        reg_scale = F.relu(5 - scales[:, :, :2] / scales[:, :, 2:]).sum() * self.scale_loss_weight
        loss = loss + reg_scale
        loss_dict.update(reg_scale=reg_scale)

        reg_rgb = ((out_part[..., :3] - skin_color.reshape(-1, 1, 1, 1, 3))*occulusion_mask.unsqueeze(-1)).abs().mean() * self.skin_loss_weight
        loss = loss + reg_rgb 
        loss_dict.update(reg_rgb=reg_rgb)

        val = torch.clamp(out_seg_part[..., :1], 1e-5, 1 - 1e-5)
        reg_opacity = (val*torch.log(val) + (1-val)*torch.log(1-val)).mean() * -1 * self.opacity_loss_weight
        loss = loss + reg_opacity
        loss_dict.update(reg_opacity=reg_opacity)

        if self.reg_loss is not None:
            reg_loss = self.reg_loss(attrmap, **kwargs)
            loss = loss + reg_loss
            loss_dict.update(reg_loss=reg_loss)
        if self.per_loss is not None:
            if self.per_loss.loss_weight > 0 and (not init):
                per_loss = self.per_loss(out_rgbs[:, 0, :, :, :3], target_rgbs[:, 0, :, :, :3])
                loss = loss + per_loss
                loss_dict.update(per_loss=per_loss)
        return out_rgbs, loss, loss_dict

    def loss_s(self, decoder, code, target_rgbs, target_segs, cameras,   
            dt_gamma=0.0, smpl_params=None, return_decoder_loss=False, scale_num_ray=1.0,
             cfg=dict(), init=False, norm=None, **kwargs):
        # target_rgbs min:0, max:1 shape: num_scenes, num_imgs, h, w, 3
        num_batch, num_imgs, h, w = target_rgbs.shape[:4]
        outputs = decoder(
            code, self.grid_size, smpl_params, cameras,
            num_imgs, dt_gamma=dt_gamma, perturb=True, return_loss=return_decoder_loss, init=init, return_norm=(norm!=None))

        out_rgbs = outputs['image']
        out_segs = outputs['segs']
        target_segs = target_segs.float()
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l3.png', (out_segs[0, 0, :, :, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l3.png', (target_segs[0, 0, :, :, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_l0.png', (out_rgbs[0, 0, :, :, 3:6].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_l0.png', (target_rgbs[0, 0, :, :, 3:6].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l0.png', (out_segs[0, 0, :, :, 1].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l0.png', (target_segs[0, 0, :, :, 1].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l1.png', (out_segs[0, 0, :, :, 2].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l1.png', (target_segs[0, 0, :, :, 2].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l2.png', (out_segs[0, 0, :, :, 3].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l2.png', (target_segs[0, 0, :, :, 3].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        # out_rgbs = outputs['image']
        
        out_offsets = outputs['offset']
        attrmap = outputs['attrmap']
        scales = outputs['scales']
        
        loss = 0
        scale = 1 - math.exp(-cfg['loss_coef'] * scale_num_ray) if 'loss_coef' in cfg else 1
        # seg_part = target_seg_part.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3).flatten(4, 5)
        seg_part = target_segs.unsqueeze(-1).expand(-1, -1, -1, -1, -1, 3).flatten(4, 5)
        pixel_loss = huber(out_rgbs, target_rgbs) * (scale * 3) * self.pixel_loss_weight# 2.5
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_l3.png', (out_rgbs[0, 0, :, :, :3].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_l3.png', (target_rgbs[0, 0, :, :, :3].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_l0.png', (out_rgbs[0, 0, :, :, 3:6].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_l0.png', (target_rgbs[0, 0, :, :, 3:6].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/seg_l0.png', (out_segs[0, 1, :, :, 1].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_seg_l0.png', (target_segs[0, 1, :, :, 1].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_l1.png', (out_rgbs[0, 0, :, :, 6:9].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_l1.png', (target_rgbs[0, 0, :, :, 6:9].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/out_l2.png', (out_rgbs[0, 0, :, :, 9:12].detach().cpu().numpy()*255).astype(np.uint8))
        # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/target_l2.png', (target_rgbs[0, 0, :, :, 9:12].detach().cpu().numpy()*255).astype(np.uint8))
        # assert False
        loss = loss + pixel_loss
        loss_dict = dict(pixel_loss=pixel_loss)
        
        seg_loss = huber(out_segs, target_segs) * scale * self.seg_loss_weight
        loss = loss + seg_loss 
        loss_dict.update(seg_loss=seg_loss)
        
        reg_offset = out_offsets * cfg['offset_weight'] 
        loss = loss + reg_offset
        loss_dict.update(reg_offset=reg_offset)

        reg_scale = F.relu(5 - scales[:, :, :2] / scales[:, :, 2:]).sum() * self.scale_loss_weight
        loss = loss + reg_scale
        loss_dict.update(reg_scale=reg_scale)

        val = torch.clamp(out_segs[..., 1:2], 1e-5, 1 - 1e-5)
        reg_opacity = (val*torch.log(val) + (1-val)*torch.log(1-val)).mean() * -1 * self.opacity_loss_weight
        loss = loss + reg_opacity
        loss_dict.update(reg_opacity=reg_opacity)

        if self.reg_loss is not None:
            reg_loss = self.reg_loss(attrmap, **kwargs)
            loss = loss + reg_loss
            loss_dict.update(reg_loss=reg_loss)
        if self.per_loss is not None:
            if self.per_loss.loss_weight > 0 and (not init):
                # torch.randint(5)
                per_loss = self.per_loss(out_rgbs[:, 0, :, :, :3], target_rgbs[:, 0, :, :, :3])
                loss = loss + per_loss
                loss_dict.update(per_loss=per_loss)
        return out_rgbs, loss, loss_dict

    # def loss_decoder(self, decoder, code, density_bitfield, cond_rays_o, cond_rays_d, cond_imgs, smpl_params=None, 
    #     dt_gamma=0.0, cfg=dict(), ortho=True, **kwargs):
    # def loss_decoder(self, decoder, code, density_bitfield, cond_imgs, cameras, smpl_params=None, 
    #     dt_gamma=0.0, cfg=dict(), ortho=True, **kwargs):
    def loss_decoder(self, decoder, code, cond_imgs, cond_segs, cameras, smpl_params=None, 
        dt_gamma=0.0, cfg=dict(), densify=False, init=False, cond_norm=None, **kwargs):
        decoder_training_prev = decoder.training
        decoder.train(True)
        # n_decoder_rays = cfg.get('n_decoder_rays', 4096)
        if smpl_params == None:
            assert False

        num_scenes, num_imgs, _, _, _ = cond_imgs.shape
        select_imgs = cond_imgs[:, 3::4]
        select_segs = cond_segs[:, 3::4]
        select_cameras = cameras[:, 3::4]
        # select_imgs = cond_imgs[:, 2::3]
        # select_segs = cond_segs[:, 2::3]
        # select_cameras = cameras[:, 2::3]
        # select_imgs = cond_imgs
        # select_segs = cond_segs
        # select_cameras = cameras
        # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/debug_fit_gt.jpg', (select_imgs[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # print(select_imgs.shape)
        # print(select_cameras.shape)
        # assert False
        if cond_norm != None:
            select_norm = cond_norm[:, 3::4]
        else:
            select_norm = None
        # select_imgs = cond_imgs
        # select_cameras = cameras
        # select_imgs, select_cameras = self.sample_imgs(cond_imgs, cameras, num_scenes, num_imgs, num_samples=1, device=code.device)

        # rays_o, rays_d, target_rgbs = self.ray_sample(
        #     cond_rays_o, cond_rays_d, cond_imgs, n_samples=n_decoder_rays, ortho=ortho)
        # print(cond_rays_o.shape[1:4].numel())
        # print(cond_rays_o.shape)
        # cond_rays_o num_scenes, num_imgs, h, w, 3
        # scale_num_ray num_imgs * h * w
        # assert False
        # out_rgbs, loss, loss_dict = self.loss(
        #     decoder, code, density_bitfield, cond_imgs, cameras,
        #     dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
        #     cfg=cfg, **kwargs)
        out_rgbs, loss, loss_dict = self.loss(
            decoder, code, select_imgs, select_segs, select_cameras, 
            dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
            cfg=cfg, init=init, norm=select_norm, **kwargs)
        # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/debug_fit_pred.jpg', (out_rgbs[0, 0].detach().cpu().numpy()*255).astype(np.uint8))
        # print(out_rgbs.shape)
        # assert False

        log_vars = dict()
        for key, val in loss_dict.items():
            log_vars.update({key: float(val)})

        decoder.train(decoder_training_prev)

        # return loss, log_vars, out_rgbs, cond_imgs
        return loss, log_vars, out_rgbs, select_imgs
    
    def loss_decoder_t(self, decoder, code, cond_imgs, cond_segs, cameras, smpl_params=None, 
        dt_gamma=0.0, cfg=dict(), densify=False, init=False, cond_norm=None, **kwargs):
        decoder_training_prev = decoder.training
        decoder.train(True)
        # n_decoder_rays = cfg.get('n_decoder_rays', 4096)
        if smpl_params == None:
            assert False

        num_scenes, num_imgs, _, _, _ = cond_imgs.shape
        select_imgs = cond_imgs[:, 3::4]
        select_segs = cond_segs[:, 3::4]
        select_cameras = cameras[:, 3::4]
        # select_imgs = cond_imgs[:, 2::3]
        # select_segs = cond_segs[:, 2::3]
        # select_cameras = cameras[:, 2::3]
        # select_imgs = cond_imgs
        # select_segs = cond_segs
        # select_cameras = cameras
        if cond_norm != None:
            select_norm = cond_norm[:, 3::4]
        else:
            select_norm = None
      
        out_rgbs, loss, loss_dict = self.loss_t(
            decoder, code, select_imgs, select_segs, select_cameras, 
            dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
            cfg=cfg, init=init, norm=select_norm, **kwargs)

        log_vars = dict()
        for key, val in loss_dict.items():
            log_vars.update({key: float(val)})

        decoder.train(decoder_training_prev)

        return loss, log_vars, out_rgbs, select_imgs

    def loss_decoder_s(self, decoder, code, cond_imgs, cond_segs, cameras, smpl_params=None, 
        dt_gamma=0.0, cfg=dict(), densify=False, init=False, cond_norm=None, **kwargs):
        decoder_training_prev = decoder.training
        decoder.train(True)
        # n_decoder_rays = cfg.get('n_decoder_rays', 4096)
        if smpl_params == None:
            assert False

        num_scenes, num_imgs, _, _, _ = cond_imgs.shape
        select_imgs = cond_imgs[:, 3::4]
        select_segs = cond_segs[:, 3::4]
        select_cameras = cameras[:, 3::4]
        # select_imgs = cond_imgs[:, 2::3]
        # select_segs = cond_segs[:, 2::3]
        # select_cameras = cameras[:, 2::3]
        # select_imgs = cond_imgs
        # select_segs = cond_segs
        # select_cameras = cameras
        if cond_norm != None:
            select_norm = cond_norm[:, 3::4]
        else:
            select_norm = None
      
        out_rgbs, loss, loss_dict = self.loss_s(
            decoder, code, select_imgs, select_segs, select_cameras, 
            dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
            cfg=cfg, init=init, norm=select_norm, **kwargs)

        log_vars = dict()
        for key, val in loss_dict.items():
            log_vars.update({key: float(val)})

        decoder.train(decoder_training_prev)

        return loss, log_vars, out_rgbs, select_imgs

    # def update_extra_state(self, decoder, code, density_grid, density_bitfield,
    #                        iter_density, smpl_params=None, density_thresh=0.01, decay=0.9, S=128):
    def update_extra_state(self, decoder, code, masks, points,
                           iter_density, smpl_params=None, density_thresh=0.01, scale_thresh=0.05, decay=0.9, S=128, offset_thresh=0.005):
        with torch.no_grad():
            device = get_module_device(self)
            # num_scenes = density_grid.size(0)
            num_scenes, num_points = masks.shape
            # tmp_grid = torch.full_like(density_grid, -1)
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module

            # densify 
            # sigmas, offset, radius, _ = decoder.point_density_decode(
            #                     points, code, smpl_params=None)
            # sigmas = sigmas.reshape(num_scenes, -1)
            # offset = offset.reshape(num_scenes, -1, 3)
            #                     # [0].reshape(num_scenes, -1)  # (num_scenes, N)
            # masks = sigmas > density_thresh
            # offset_dist = torch.sqrt((offset**2).sum(2))
            # ############
            # # offset_dist (min:1.28e-5, max:0.02-0.035, mean:0.00218, median:0.00158-0.00178)
            # # radius(min: 9.31e-7, max: 0.99, mean: 0.1613-0.1729, median:0.022-0.029)
            # # print(points.shape)
            # # print(offset.shape)
            # # print(sigmas.shape)
            # # assert False
            
            # try:
            #     points = points + offset
            # except:
            #     print(points.shape)
            #     print(offset.shape)
            #     assert False
            # if (torch.logical_and(masks, offset_dist<offset_thresh)).sum() > num_scenes * 100000:
            #     return masks, points
            # else:
            #     points_new = []
            #     select_masks = torch.logical_and(masks, offset_dist>offset_thresh)
            #     for sigma, point, mask, select_mask in zip(sigmas, points, masks, select_masks):
            #         num_vis = mask.sum()
            #         select_pcd = torch.cat([point[select_mask], point[mask]], dim=0)
            #         select_idx = torch.randint(select_pcd.shape[0], (num_points-num_vis, ))
            #         point[~mask] = select_pcd[select_idx] + (torch.rand((num_points-num_vis, 3), device=device) * 0.016 - 0.008)
            #         points_new.append(point)
            #     points_new = torch.stack(points_new, dim=0)
            #     return masks, points_new
            ## prune
            sigmas, _, _, _ = decoder.point_density_decode(
                                points, code, smpl_params=None)
            sigmas = sigmas.reshape(num_scenes, -1)
            masks = sigmas > density_thresh
            
            # points_new = []
            # for point, mask in zip(points, masks):
            #     point[~mask] = 0
            #     points_new.append(point)
            # points_new = torch.stack(points_new, dim=0)
            return masks, points


    def get_density(self, decoder, code, cfg=dict()):
        # density_thresh = cfg.get('density_thresh', 0.01)
        density_thresh = cfg.get('density_thresh', 0.005)
        # density_step = cfg.get('density_step', 8)
        density_step = cfg.get('density_step', 1)
        num_scenes = code.size(0)
        try:
            num_init = decoder.module.num_init
        except:
            num_init = decoder.num_init
        device = code.device
        masks = self.get_init_mask_(num_scenes, num_init, device)
        try:
            points = self.get_init_points_(num_scenes, decoder.module.init_pcd, device)
        except:
            points = self.get_init_points_(num_scenes, decoder.init_pcd, device)
        # points_out = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_out.npy')).unsqueeze(0).expand(num_scenes, -1, -1).to(points.device)
        # density_grid = self.get_init_density_grid(num_scenes, device)
        # density_bitfield = self.get_init_density_bitfield(num_scenes, device)
        # for i in range(density_step):
            # self.update_extra_state(decoder, code, density_grid, density_bitfield, i,
            #                         density_thresh=density_thresh, decay=1.0)
        masks, points = self.update_extra_state(decoder, code, masks, points, 0,
                                density_thresh=density_thresh, decay=1.0)
        # from pytorch3d import ops
        # select_pcd = points[0][masks[0]]
        # dist_sq, idx, neighbors = ops.knn_points(points_out[0].float().unsqueeze(0), select_pcd.float()[::10].unsqueeze(0), K=1)
        # mask_new = dist_sq < 0.02 ** 2
        # print(mask_new.shape)
        # print(mask_new.sum())
        # assert False
        return masks, points

    # def inverse_code(self, decoder, cond_imgs, cameras, cond_rays_o, cond_rays_d, dt_gamma=0, smpl_params=None, cfg=dict(),
    #                  code_=None, density_grid=None, density_bitfield=None, iter_density=None,
    #                  code_optimizer=None, code_scheduler=None,
    #                  prior_grad=None, show_pbar=False, ortho=True):
    def sample_imgs(self, cond_imgs, cameras, num_scenes, num_imgs, num_samples, device):
        sample_inds = [torch.randperm(num_imgs, device=device)[:num_samples] for _ in range(num_scenes)]
        sample_inds = torch.stack(sample_inds, dim=0)
        scene_arange = torch.arange(num_scenes, device=device)[:, None]
        cond_imgs_select = cond_imgs[scene_arange, sample_inds]
        cameras_select = cameras[scene_arange, sample_inds]
        return cond_imgs_select, cameras_select

    def inverse_code(self, decoder, cond_imgs, cond_segs, cameras, dt_gamma=0, smpl_params=None, cfg=dict(),
                     code_=None, iter_density=None, init=None, cond_norm=None,
                     code_optimizer=None, code_scheduler=None,
                     prior_grad=None, show_pbar=False, densify=False):
        """
        Obtain scene codes via optimization-based inverse rendering.
        """
        device = get_module_device(self)
        decoder_training_prev = decoder.training
        decoder.train(True)
        assert smpl_params != None

        with module_requires_grad(decoder, False):
            n_inverse_steps = cfg.get('n_inverse_steps', 4)
            # n_inverse_rays = cfg.get('n_inverse_rays', 4096)

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # assert n_inverse_steps * 4 <= num_imgs
            # if n_inverse_steps * 4 > num_imgs:
            #     assert False
            num_scene_pixels = num_imgs * h * w
            # raybatch_inds, num_raybatch = self.get_raybatch_inds(cond_imgs, n_inverse_rays)
            # torch.cuda.synchronize()
            # start_time = time.time()
            if code_ is None:
                code_ = self.get_init_code_(num_scenes, device=device)
            # if density_grid is None:
            #     density_grid = self.get_init_density_grid(num_scenes, device)
            # if density_bitfield is None:
            #     density_bitfield = self.get_init_density_bitfield(num_scenes, device)
            if iter_density is None:
                iter_density = 0

            if code_optimizer is None:
                assert code_scheduler is None
                code_optimizer = self.build_optimizer(code_, cfg)
            if code_scheduler is None:
                code_scheduler = self.build_scheduler(code_optimizer, cfg)

            assert n_inverse_steps > 0
            if show_pbar:
                pbar = mmcv.ProgressBar(n_inverse_steps)
            # print(f'init_code_time:{init_code_time-start_time}')
            for inverse_step_id in range(n_inverse_steps):
                code = self.code_activation(
                    torch.stack(code_, dim=0) if isinstance(code_, list)
                    else code_)
                
                # select_imgs, select_cameras = self.sample_imgs(cond_imgs, cameras, num_scenes, num_imgs, num_samples=1, device=device)
                # print(cond_imgs.shape)
                # assert False
                select_imgs = cond_imgs[:, inverse_step_id::4]
                select_segs = cond_segs[:, inverse_step_id::4]
                select_cameras = cameras[:, inverse_step_id::4]
                # select_imgs = cond_imgs
                # select_segs = cond_segs
                # select_cameras = cameras
                # select_imgs = cond_imgs[:, inverse_step_id::3]
                # select_segs = cond_segs[:, inverse_step_id::3]
                # select_cameras = cameras[:, inverse_step_id::3]
                # select_imgs = cond_imgs[:, -8:-4]
                # select_segs = cond_segs[:, -8:-4]
                # select_cameras = cameras[:, -8:-4]
                # select_imgs = cond_imgs
                # select_cameras = cameras
                if cond_norm != None:
                    select_norm = cond_norm[:, inverse_step_id::4]
                else:
                    select_norm = None
                # torch.cuda.synchronize()
                # init_code_time = time.time()
                # select_imgs = cond_imgs
                # select_cameras = cameras
                      
                # if (inverse_step_id % self.update_extra_interval == 0) and densify:
                #     masks, points = self.update_extra_state(decoder, code, masks, points,
                #                             iter_density, smpl_params=None, density_thresh=cfg.get('density_thresh', 0.01))

                # inds = raybatch_inds[inverse_step_id % num_raybatch] if raybatch_inds is not None else None
                # rays_o, rays_d, target_rgbs = self.ray_sample(
                #     cond_rays_o, cond_rays_d, cond_imgs, n_inverse_rays, sample_inds=inds)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code, density_bitfield,
                #     cond_imgs, cameras, dt_gamma, smpl_params=smpl_params, scale_num_ray=num_scene_pixels,
                #     cfg=cfg)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code,
                #     select_imgs, select_cameras, points, masks, dt_gamma, smpl_params=smpl_params, scale_num_ray=num_scene_pixels,
                #     cfg=cfg, stage2=densify, init=init)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code,
                #     select_imgs, select_segs, select_cameras, dt_gamma, smpl_params=smpl_params, scale_num_ray=cond_imgs.shape[1:4].numel(),
                #     cfg=cfg, init=init, norm=select_norm)
                out_rgbs, loss, loss_dict = self.loss(
                    decoder, code, select_imgs, select_segs, select_cameras, 
                    dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
                    cfg=cfg, init=init)
                # torch.cuda.synchronize()
                # loss_time = time.time()
                # print(f'loss_time:{loss_time-init_code_time}')
                
                if prior_grad is not None:
                    if isinstance(code_, list):
                        for code_single_, prior_grad_single in zip(code_, prior_grad):
                            code_single_.grad.copy_(prior_grad_single)
                    else:
                        code_.grad.copy_(prior_grad)
                else:
                    if isinstance(code_optimizer, list):
                        for code_optimizer_single in code_optimizer:
                            code_optimizer_single.zero_grad()
                    else:
                        code_optimizer.zero_grad()

                loss.backward()

                if isinstance(code_optimizer, list):
                    for code_optimizer_single in code_optimizer:
                        code_optimizer_single.step()
                else:
                    code_optimizer.step()

                if code_scheduler is not None:
                    if isinstance(code_scheduler, list):
                        for code_scheduler_single in code_scheduler:
                            code_scheduler_single.step()
                    else:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()
                
            #     torch.cuda.synchronize()
            #     run_time = time.time()
            #     print(f'run_time:{run_time-loss_time}')
            # assert False
        decoder.train(decoder_training_prev)
        return code.detach(), loss, loss_dict

    def inverse_code_t(self, decoder, cond_imgs, cond_segs, cameras, dt_gamma=0, smpl_params=None, cfg=dict(),
                     code_=None, iter_density=None, init=None, cond_norm=None,
                     code_optimizer=None, code_scheduler=None,
                     prior_grad=None, show_pbar=False, densify=False):
        """
        Obtain scene codes via optimization-based inverse rendering.
        """
        device = get_module_device(self)
        decoder_training_prev = decoder.training
        decoder.train(True)
        assert smpl_params != None

        with module_requires_grad(decoder, False):
            n_inverse_steps = cfg.get('n_inverse_steps', 4)
            # n_inverse_rays = cfg.get('n_inverse_rays', 4096)

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # assert n_inverse_steps * 4 <= num_imgs
            # if n_inverse_steps * 4 > num_imgs:
            #     assert False
            num_scene_pixels = num_imgs * h * w
            # raybatch_inds, num_raybatch = self.get_raybatch_inds(cond_imgs, n_inverse_rays)
            # torch.cuda.synchronize()
            # start_time = time.time()
            if code_ is None:
                code_ = self.get_init_code_(num_scenes, device=device)
            # if density_grid is None:
            #     density_grid = self.get_init_density_grid(num_scenes, device)
            # if density_bitfield is None:
            #     density_bitfield = self.get_init_density_bitfield(num_scenes, device)
            if iter_density is None:
                iter_density = 0

            if code_optimizer is None:
                assert code_scheduler is None
                code_optimizer = self.build_optimizer(code_, cfg)
            if code_scheduler is None:
                code_scheduler = self.build_scheduler(code_optimizer, cfg)

            assert n_inverse_steps > 0
            if show_pbar:
                pbar = mmcv.ProgressBar(n_inverse_steps)
            # print(f'init_code_time:{init_code_time-start_time}')
            for inverse_step_id in range(n_inverse_steps):
                code = self.code_activation(
                    torch.stack(code_, dim=0) if isinstance(code_, list)
                    else code_)
                
                # select_imgs, select_cameras = self.sample_imgs(cond_imgs, cameras, num_scenes, num_imgs, num_samples=1, device=device)
                # print(cond_imgs.shape)
                # assert False
                select_imgs = cond_imgs[:, inverse_step_id::4]
                select_segs = cond_segs[:, inverse_step_id::4]
                select_cameras = cameras[:, inverse_step_id::4]
                # select_imgs = cond_imgs[:, inverse_step_id::3]
                # select_segs = cond_segs[:, inverse_step_id::3]
                # select_cameras = cameras[:, inverse_step_id::3]
                # select_imgs = cond_imgs
                # select_segs = cond_segs
                # select_cameras = cameras
                # select_imgs = cond_imgs
                # select_cameras = cameras
                if cond_norm != None:
                    select_norm = cond_norm[:, inverse_step_id::4]
                else:
                    select_norm = None
                # torch.cuda.synchronize()
                # init_code_time = time.time()
                # select_imgs = cond_imgs
                # select_cameras = cameras
                      
                # if (inverse_step_id % self.update_extra_interval == 0) and densify:
                #     masks, points = self.update_extra_state(decoder, code, masks, points,
                #                             iter_density, smpl_params=None, density_thresh=cfg.get('density_thresh', 0.01))

                # inds = raybatch_inds[inverse_step_id % num_raybatch] if raybatch_inds is not None else None
                # rays_o, rays_d, target_rgbs = self.ray_sample(
                #     cond_rays_o, cond_rays_d, cond_imgs, n_inverse_rays, sample_inds=inds)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code, density_bitfield,
                #     cond_imgs, cameras, dt_gamma, smpl_params=smpl_params, scale_num_ray=num_scene_pixels,
                #     cfg=cfg)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code,
                #     select_imgs, select_cameras, points, masks, dt_gamma, smpl_params=smpl_params, scale_num_ray=num_scene_pixels,
                #     cfg=cfg, stage2=densify, init=init)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code,
                #     select_imgs, select_segs, select_cameras, dt_gamma, smpl_params=smpl_params, scale_num_ray=cond_imgs.shape[1:4].numel(),
                #     cfg=cfg, init=init, norm=select_norm)
                out_rgbs, loss, loss_dict = self.loss_t(
                    decoder, code, select_imgs, select_segs, select_cameras, 
                    dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
                    cfg=cfg, init=init)
                # torch.cuda.synchronize()
                # loss_time = time.time()
                # print(f'loss_time:{loss_time-init_code_time}')
                
                if prior_grad is not None:
                    if isinstance(code_, list):
                        for code_single_, prior_grad_single in zip(code_, prior_grad):
                            code_single_.grad.copy_(prior_grad_single)
                    else:
                        code_.grad.copy_(prior_grad)
                else:
                    if isinstance(code_optimizer, list):
                        for code_optimizer_single in code_optimizer:
                            code_optimizer_single.zero_grad()
                    else:
                        code_optimizer.zero_grad()

                loss.backward()

                if isinstance(code_optimizer, list):
                    for code_optimizer_single in code_optimizer:
                        code_optimizer_single.step()
                else:
                    code_optimizer.step()

                if code_scheduler is not None:
                    if isinstance(code_scheduler, list):
                        for code_scheduler_single in code_scheduler:
                            code_scheduler_single.step()
                    else:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()
                
            #     torch.cuda.synchronize()
            #     run_time = time.time()
            #     print(f'run_time:{run_time-loss_time}')
            # assert False
        decoder.train(decoder_training_prev)
        return code.detach(), loss, loss_dict

    def inverse_code_s(self, decoder, cond_imgs, cond_segs, cameras, dt_gamma=0, smpl_params=None, cfg=dict(),
                     code_=None, iter_density=None, init=None, cond_norm=None,
                     code_optimizer=None, code_scheduler=None,
                     prior_grad=None, show_pbar=False, densify=False):
        """
        Obtain scene codes via optimization-based inverse rendering.
        """
        device = get_module_device(self)
        decoder_training_prev = decoder.training
        decoder.train(True)
        assert smpl_params != None

        with module_requires_grad(decoder, False):
            n_inverse_steps = cfg.get('n_inverse_steps', 4)
            # n_inverse_rays = cfg.get('n_inverse_rays', 4096)

            num_scenes, num_imgs, h, w, _ = cond_imgs.size()
            # assert n_inverse_steps * 4 <= num_imgs
            # if n_inverse_steps * 4 > num_imgs:
            #     assert False
            num_scene_pixels = num_imgs * h * w
            # raybatch_inds, num_raybatch = self.get_raybatch_inds(cond_imgs, n_inverse_rays)
            # torch.cuda.synchronize()
            # start_time = time.time()
            if code_ is None:
                code_ = self.get_init_code_(num_scenes, device=device)
            # if density_grid is None:
            #     density_grid = self.get_init_density_grid(num_scenes, device)
            # if density_bitfield is None:
            #     density_bitfield = self.get_init_density_bitfield(num_scenes, device)
            if iter_density is None:
                iter_density = 0

            if code_optimizer is None:
                assert code_scheduler is None
                code_optimizer = self.build_optimizer(code_, cfg)
            if code_scheduler is None:
                code_scheduler = self.build_scheduler(code_optimizer, cfg)

            assert n_inverse_steps > 0
            if show_pbar:
                pbar = mmcv.ProgressBar(n_inverse_steps)
            # print(f'init_code_time:{init_code_time-start_time}')
            for inverse_step_id in range(n_inverse_steps):
                code = self.code_activation(
                    torch.stack(code_, dim=0) if isinstance(code_, list)
                    else code_)
                
                # select_imgs, select_cameras = self.sample_imgs(cond_imgs, cameras, num_scenes, num_imgs, num_samples=1, device=device)
                # print(cond_imgs.shape)
                # assert False
                select_imgs = cond_imgs[:, inverse_step_id::4]
                select_segs = cond_segs[:, inverse_step_id::4]
                select_cameras = cameras[:, inverse_step_id::4]
                # select_imgs = cond_imgs[:, inverse_step_id::3]
                # select_segs = cond_segs[:, inverse_step_id::3]
                # select_cameras = cameras[:, inverse_step_id::3]
                # select_imgs = cond_imgs
                # select_segs = cond_segs
                # select_cameras = cameras
                # select_imgs = cond_imgs
                # select_cameras = cameras
                if cond_norm != None:
                    select_norm = cond_norm[:, inverse_step_id::4]
                else:
                    select_norm = None
                # torch.cuda.synchronize()
                # init_code_time = time.time()
                # select_imgs = cond_imgs
                # select_cameras = cameras
                      
                # if (inverse_step_id % self.update_extra_interval == 0) and densify:
                #     masks, points = self.update_extra_state(decoder, code, masks, points,
                #                             iter_density, smpl_params=None, density_thresh=cfg.get('density_thresh', 0.01))

                # inds = raybatch_inds[inverse_step_id % num_raybatch] if raybatch_inds is not None else None
                # rays_o, rays_d, target_rgbs = self.ray_sample(
                #     cond_rays_o, cond_rays_d, cond_imgs, n_inverse_rays, sample_inds=inds)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code, density_bitfield,
                #     cond_imgs, cameras, dt_gamma, smpl_params=smpl_params, scale_num_ray=num_scene_pixels,
                #     cfg=cfg)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code,
                #     select_imgs, select_cameras, points, masks, dt_gamma, smpl_params=smpl_params, scale_num_ray=num_scene_pixels,
                #     cfg=cfg, stage2=densify, init=init)
                # out_rgbs, loss, loss_dict = self.loss(
                #     decoder, code,
                #     select_imgs, select_segs, select_cameras, dt_gamma, smpl_params=smpl_params, scale_num_ray=cond_imgs.shape[1:4].numel(),
                #     cfg=cfg, init=init, norm=select_norm)
                out_rgbs, loss, loss_dict = self.loss_s(
                    decoder, code, select_imgs, select_segs, select_cameras, 
                    dt_gamma, smpl_params=smpl_params, return_decoder_loss=True, scale_num_ray=cond_imgs.shape[1:4].numel(),
                    cfg=cfg, init=init)
                # torch.cuda.synchronize()
                # loss_time = time.time()
                # print(f'loss_time:{loss_time-init_code_time}')
                
                if prior_grad is not None:
                    if isinstance(code_, list):
                        for code_single_, prior_grad_single in zip(code_, prior_grad):
                            code_single_.grad.copy_(prior_grad_single)
                    else:
                        code_.grad.copy_(prior_grad)
                else:
                    if isinstance(code_optimizer, list):
                        for code_optimizer_single in code_optimizer:
                            code_optimizer_single.zero_grad()
                    else:
                        code_optimizer.zero_grad()

                loss.backward()

                if isinstance(code_optimizer, list):
                    for code_optimizer_single in code_optimizer:
                        code_optimizer_single.step()
                else:
                    code_optimizer.step()

                if code_scheduler is not None:
                    if isinstance(code_scheduler, list):
                        for code_scheduler_single in code_scheduler:
                            code_scheduler_single.step()
                    else:
                        code_scheduler.step()

                if show_pbar:
                    pbar.update()
                
            #     torch.cuda.synchronize()
            #     run_time = time.time()
            #     print(f'run_time:{run_time-loss_time}')
            # assert False
        decoder.train(decoder_training_prev)
        return code.detach(), loss, loss_dict

    # def render(self, decoder, code, density_bitfield, h, w, intrinsics, poses, smpl_params, cfg=dict()):
    def render(self, decoder, code, h, w, intrinsics, poses, smpl_params, cfg=dict(), mask=None, return_norm=False, return_viz=False):
        decoder_training_prev = decoder.training
        decoder.train(False)
        # dt_gamma_scale = cfg.get('dt_gamma_scale', 0.0)
        # (num_scenes,)
        # dt_gamma = dt_gamma_scale * 2 / (intrinsics[..., 0] + intrinsics[..., 1]).mean(dim=-1)
        dt_gamma = 0.0
        # rays_o, rays_d = get_cam_rays(poses, intrinsics, h, w)
        num_scenes, num_imgs, _, _ = poses.size()
        # num_scenes, num_imgs, h, w, _ = rays_o.size()

        # these two lines are for pointrendering method camera
        # poses[:,:,:,:2] *= (-1)
        # cameras = [FoVPerspectiveCameras(fov=46.2, R=poses[i,:,:3,:3], T=poses[i,0,:3, 3].expand(num_imgs, -1), device=poses.device) for i in range(num_scenes)]
        # cameras = [[intrinsics[i], poses[i]] for i in range(num_scenes)]
        # cameras = [intrinsics, poses]
        # cameras = [poses[i, :] for i in range(num_scenes)]
        # assert False

        # rays_o = rays_o.reshape(num_scenes, num_imgs * h * w, 3)
        # rays_d = rays_d.reshape(num_scenes, num_imgs * h * w, 3)
        # max_render_rays = cfg.get('max_render_rays', -1)
        # if 0 < max_render_rays < rays_o.size(1):
        #     rays_o = rays_o.split(max_render_rays, dim=1)
        #     rays_d = rays_d.split(max_render_rays, dim=1)
        # else:
        #     rays_o = [rays_o]
        #     rays_d = [rays_d]

        # out_image = []
        # out_depth = []
        # for rays_o_single, rays_d_single, smpl_param_single in zip(rays_o, rays_d, smpl_params):
        cameras = torch.cat([intrinsics, poses.reshape(num_scenes, num_imgs, -1)], dim=-1)
        outputs = decoder(
            code, self.grid_size, smpl_params, cameras,
            num_imgs, mask=mask, dt_gamma=dt_gamma, perturb=False)
        out_image = outputs['image']
        out_seg = outputs['segs']
        out_vizmask = outputs['viz_masks']
        # out_coloruv = outputs['colormap']
        
        if return_norm:
            # out_norm = outputs['norm_maps']
            out_norm = outputs['norm']
            out_norm = out_norm.reshape(num_scenes, num_imgs, h, w, 3)
        else:
            out_norm = None
            # weights = torch.stack(outputs['weights_sum'], dim=0) if num_scenes > 1 else outputs['weights_sum'][0]
            # rgbs = (torch.stack(outputs['image'], dim=0) if num_scenes > 1 else outputs['image'][0]) \
            #        + self.bg_color * (1 - weights.unsqueeze(-1))
            # rgbs = torch.stack(outputs['image'], dim=0) if num_scenes > 1 else outputs['image'][0]
            # depth = torch.stack(outputs['depth'], dim=0) if num_scenes > 1 else outputs['depth'][0]
            # out_image.append(rgbs)
            # out_depth.append(depth)
        # out_image = torch.cat(out_image, dim=1) if len(out_image) > 1 else out_image[0]
        # out_depth = torch.cat(out_depth, dim=1) if len(out_depth) > 1 else out_depth[0]
        out_image = out_image.reshape(num_scenes, num_imgs, h, w, -1)
        out_seg = out_seg.reshape(num_scenes, num_imgs, h, w, -1)
        # out_depth = out_depth.reshape(num_scenes, num_imgs, h, w)

        decoder.train(decoder_training_prev)
        if return_viz:
            return out_image, out_norm, out_vizmask, out_seg
        else:
            return out_image, out_norm, out_seg

    def exmesh_step(self, data, viz_dir=None, **kwargs):
        scene_name = data['scene_name']  # (num_scenes,)
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder
        # num_scenes, num_imgs, _, _ = test_poses.size()
        with torch.no_grad():
            if 'code' in data:
                code = self.load_scene(data, load_density=True)
            elif 'cond_imgs' in data:
                raise AttributeError
            else:
                code = self.val_uncond(data, **kwargs)
            num_scenes = 1
            test_smpl_param = data['test_smpl_param'].to(code.device)
            xyzs, sigmas, rgbs, _, _, tfs, rot, _, _ = decoder.extract_pcd(code[0:1], test_smpl_param[0:1], init=False)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(decoder.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
            # remain_xyzs = xyzs[sigmas>0.05]
            # remain_xyzs = xyzs[0, 34858+21424:, 0]
            # remain_norms = normals[0, 34858+21424:]
            # remain_rgbs = rgbs[0, 34858+21424:]
            # remain_xyzs = xyzs[0, :34858, 0]
            remain_xyzs = xyzs[0, 34858+21424+11056:, 0]
            # remain_sigmas = sigmas[0, :34858, 0]
            remain_sigmas = sigmas[0, 34858+21424+11056:, 0]
            remain_xyzs = remain_xyzs[remain_sigmas>0.1]
            # remain_norms = normals[0, :34858]
            remain_norms = normals[0, 34858+21424+11056:]
            remain_norms = remain_norms[remain_sigmas>0.1]
            # remain_rgbs = rgbs[0, :34858]
            remain_rgbs = rgbs[0, 34858+21424+11056:]
            remain_rgbs = remain_rgbs[remain_sigmas>0.1]
            pcd = o3d.cuda.pybind.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(remain_xyzs.detach().cpu().numpy())
            pcd.normals = o3d.utility.Vector3dVector(remain_norms.detach().cpu().numpy())
            pcd.colors = o3d.utility.Vector3dVector(remain_rgbs.detach().cpu().numpy())
            # pcd = o3d.t.geometry.PointCloud(o3c.Tensor(remain_xyzs.detach().cpu().numpy()), device=o3c.Device("CUDA:0"))
            # print(type(pcd))
            mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

        # def udf_func(input_pcd, surface_pcd=remain_xyzs):
        #     dists, _, _ = knn_points(input_pcd.unsqueeze(0), surface_pcd.unsqueeze(0))
        #     p = dists.sqrt().squeeze()
        #     return p
        # # from Drapenet https://github.com/liren2515/DrapeNet/blob/main/encdec/export_meshes.py#L88
        # v, t = get_mesh_from_udf(
        #         udf_func,
        #         coords_range=(-1, 1),
        #         max_dist=0.2,
        #         N=512,
        #         max_batch=2**16,
        #         differentiable=False,
        #     )
        # print(v.shape)
        # print(t.shape)
        # extract_mesh = trimesh.Trimesh(vertices=v.detach().cpu(), feces=t.detach().cpu())
        # extract_mesh.export('/mnt/sdb/zwt/LayerAvatar/debug/extract_mesh.obj')
        # assert False
        # pred_mesh_o3d = get_o3d_mesh_from_tensors(v, t)
            o3d.io.write_triangle_mesh('/mnt/sdb/zwt/LayerAvatar/debug/extract_mesh_up.obj', mesh)
        assert False
        # mesh_path = get_out_dir() / f"meshes_test/{item_ids[i]}.obj"
        # mesh_path.parent.mkdir(exist_ok=True, parents=True)
        # o3d.io.write_triangle_mesh(str(mesh_path), pred_mesh_o3d)
        return 

    def load_pose(self, path):
        with open(path, 'rb') as f:
            pose_param = json.load(f)
        w2c = np.array(pose_param['cam_param'], dtype=np.float32).reshape(36,4,4)
        cam_center = w2c[:, :3, 3]
        c2w = np.linalg.inv(w2c)
        # pose[:,:2] *= -1
        # pose = np.loadtxt(path, dtype=np.float32, delimiter=' ').reshape(9, 4)
        c2w = torch.from_numpy(c2w)
        cam_to_ndc = torch.cat([c2w[:, :3, :3], c2w[:, :3, 3:]], dim=-1)
        pose = torch.cat([cam_to_ndc, cam_to_ndc.new_tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(36, -1, -1)], dim=-2)

        return [pose, torch.from_numpy(cam_center)]
    # def eval_and_viz(self, data, decoder, code, density_bitfield, viz_dir=None, cfg=dict(), ortho=True):
    def eval_and_viz(self, data, decoder, code, viz_dir=None, cfg=dict(), ortho=True, recon=False, return_norm=False):
        # s_time = time.time()
        scene_name = data['scene_name']  # (num_scenes,)
        if recon:
            # device = get_module_device(self)
            # path = '/mnt/sdb/zwt/LayerAvatar/data/cam_36.json'
            # cam_poses = self.load_pose(path)
            # data['cond_poses'] = cam_poses[0].unsqueeze(0).to(device)
            # data['cond_intrinsics'] = cam_poses[1].unsqueeze(0).to(device)
            test_intrinsics = data['cond_intrinsics'][:, 0::9]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['cond_poses'][:, 0::9]
            # test_intrinsics = data['cond_intrinsics'][:, 1::6]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses'][:, 1::6]
            # test_intrinsics = data['cond_intrinsics'][:, 1::6]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses'][:, 1::6]
            # test_smpl_param = data['cond_smpl_param']
            # test_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses']
            test_smpl_param = data['cond_smpl_param']
        else:
            # test_intrinsics = data['test_intrinsics'][:,47:52]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['test_poses'][:,47:52]
            # test_intrinsics = data['test_intrinsics'][:, 1::3]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['test_poses'][:, 1::3]
            # test_intrinsics = data['test_intrinsics'][:, 1::9]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['test_poses'][:, 1::9]
            test_intrinsics = data['test_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['test_poses']
            test_smpl_param = data['test_smpl_param']
        
        # if test_smpl_param.shape[1] > 1:
        #     num_scenes, num_imgs, _ = test_smpl_param.size()
        # else:
        num_scenes, num_imgs, _, _ = test_poses.size()
        # s2_time = time.time()
        # print(s2_time-s_time)
        if 'test_imgs' in data and not cfg.get('skip_eval', False):
            test_imgs = data['test_imgs']  # (num_scenes, num_imgs, h, w, 3)
            _, _, h, w, _ = test_imgs.size()
            test_img_paths = data['test_img_paths']  # (num_scenes, (num_imgs,))
            target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, -1, h, w)[:, :3]
            # print(target_imgs.shape)
            # target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, 4, h, w)[:, :3]
        else:
            # this one
            test_imgs = test_img_paths = target_imgs = None
            h, w = cfg['img_size']
        # image, depth = self.render(
        #     decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg)
        # image = self.render(
        #     decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg)
        if test_smpl_param.dim() == 3:
            image = []
            segs = []
            for num in range(test_smpl_param.shape[1]):
                image_single, norm_map, seg_imgs = self.render(
                    decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param[:, num], cfg=cfg, return_norm=return_norm)
                image.append(image_single)
                segs.append(seg_imgs)
            image = torch.cat(image, dim=1)
            seg_imgs = torch.cat(segs, dim=1)
        elif test_smpl_param.dim() == 2:
            # print(code.shape)
            image, norm_map, seg_imgs = self.render(
                decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg, return_norm=return_norm)
        else:
            assert False

        # num_imgs = 50
        # pred_imgs = image.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
        # num_imgs = 50
        pred_imgs = image.reshape(num_scenes*num_imgs, h, w, 6, 3).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 3, h, 6*w).clamp(min=0, max=1)
        # pred_imgs = image.reshape(num_scenes*num_imgs, h, w, 5, 3).permute(0, 4, 1, 3, 2)[..., 256:256+512].reshape(num_scenes*num_imgs, 3, h, w//2*5).clamp(min=0, max=1)
        # pred_imgs = image.reshape(num_scenes*num_imgs, h, w*6, 3).permute(0, 3, 1, 2).clamp(min=0, max=1)
        pred_imgs = torch.round(pred_imgs * 255) / 255

        # alpha_imgs = torch.cat([seg_imgs[..., :1], seg_imgs[..., 4:]], dim=-1)
        # alpha_imgs = alpha_imgs.reshape(num_scenes*num_imgs, h, w, 6, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 6*w).clamp(min=0, max=1)
        # # alpha_imgs = alpha_imgs.reshape(num_scenes*num_imgs, h, w, 5, 1).permute(0, 4, 1, 3, 2)[..., 256:256+512].reshape(num_scenes*num_imgs, 1, h, w//2*5).clamp(min=0, max=1)
        # alpha_imgs = torch.round(alpha_imgs * 255) / 255

        # seg_imgs = seg_imgs.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, -1, h, w)[:, :3]
        # seg_imgs = seg_imgs.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, -1, h, w)
        # seg_imgs = seg_imgs.reshape(num_scenes*num_imgs, h, w, 6, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 6*w).clamp(min=0, max=1)
        # seg_imgs = seg_imgs[..., 3:].reshape(num_scenes*num_imgs, h, w, 5, 1).permute(0, 4, 1, 3, 2)[..., 256:256+512].reshape(num_scenes*num_imgs, 1, h, w//2*5).clamp(min=0, max=1)
        seg_imgs = seg_imgs[..., 3:].reshape(num_scenes*num_imgs, h, w, 6, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 6*w).clamp(min=0, max=1)
        seg_imgs = torch.round(seg_imgs * 255) / 255

        # coloruv = torch.round(coloruv * 255) / 255
        
        if return_norm:
            # norm_map = norm_map * 0.5 + 0.5
            pred_norms = norm_map.permute(0, 1, 4, 2, 3).reshape(
            num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
            pred_norms = torch.round(pred_norms * 255) / 255
        # s3_time = time.time()
        # print(s3_time-s2_time)
        if test_imgs is not None:
            test_psnr = eval_psnr(pred_imgs[..., :1], target_imgs)
            test_ssim = eval_ssim_skimage(pred_imgs_eval, target_imgs, data_range=1)
            log_vars = dict(test_psnr=float(test_psnr.mean()),
                            test_ssim=float(test_ssim.mean()))
            if self.lpips is not None:
                if len(self.lpips) == 0:
                    lpips_eval = lpips.LPIPS(
                        net='vgg', eval_mode=True, pnet_tune=False).to(pred_imgs.device)
                    self.lpips.append(lpips_eval)
                test_lpips = []
                for pred_imgs_batch, target_imgs_batch in zip(
                        pred_imgs.split(LPIPS_BS, dim=0), target_imgs.split(LPIPS_BS, dim=0)):
                    test_lpips.append(self.lpips[0](pred_imgs_batch * 2 - 1, target_imgs_batch * 2 - 1).flatten())
                test_lpips = torch.cat(test_lpips, dim=0)
                log_vars.update(test_lpips=float(test_lpips.mean()))
            else:
                test_lpips = [math.nan for _ in range(num_scenes * num_imgs)]
        else:
            # this one
            log_vars = dict()
        # s4_time = time.time()
        # print(s4_time-s3_time)
        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)
        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            output_viz = torch.round(pred_imgs.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, 6*w, 3)
            # output_viz = torch.round(torch.cat([pred_imgs.permute(0, 2, 3, 1), alpha_imgs.permute(0, 2, 3, 1)], dim=-1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w*6, 4)
            # output_viz = torch.round(torch.cat([pred_imgs.permute(0, 2, 3, 1), alpha_imgs.permute(0, 2, 3, 1)], dim=-1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w//2*5, 4)
            # output_viz = torch.round(pred_imgs.permute(0, 2, 3, 1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            # output_viz = torch.round(torch.cat([pred_imgs.permute(0, 2, 3, 1), seg_imgs[:, :1].permute(0, 2, 3, 1)], dim=-1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 4)
            # output_viz = torch.round(torch.cat([pred_imgs.permute(0, 2, 3, 1), seg_imgs[:, :1].permute(0, 2, 3, 1)], dim=-1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 4)
            # output_viz = torch.round(torch.cat([pred_imgs.permute(0, 2, 3, 1), seg_imgs[:, :1].permute(0, 2, 3, 1)], dim=-1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w*3, 4)
            output_seg_viz = torch.round(seg_imgs.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, 6*w)
            # output_seg_viz = torch.round(seg_imgs.permute(0, 2, 3, 1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w)
            # output_seg_viz = torch.round(seg_imgs.permute(0, 2, 3, 1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w//2*5)
            # output_coloruv_viz = torch.round(coloruv.permute(0, 2, 3, 1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, 128, 384, 3)
            # output_coloruv_viz = output_coloruv_viz[:, ::-1]
            if return_norm:
                output_norm_viz = torch.round(pred_norms.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            if test_imgs is not None:
                real_imgs_viz = (target_imgs.permute(0, 2, 3, 1) * 255).to(
                    torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 4)[..., :3]
                output_viz = np.concatenate([real_imgs_viz, output_viz], axis=-2)

            for scene_id, scene_name_single in enumerate(scene_name):
                for img_id in range(num_imgs):
                    if test_img_paths is not None:
                        base_name = 'scene_' + scene_name_single + '_' + os.path.splitext(
                            os.path.basename(test_img_paths[scene_id][img_id]))[0]
                        name = base_name + '_psnr{:02.1f}_ssim{:.2f}_lpips{:.3f}.png'.format(
                            test_psnr[scene_id * num_imgs + img_id],
                            test_ssim[scene_id * num_imgs + img_id],
                            test_lpips[scene_id * num_imgs + img_id])
                        existing_files = glob(os.path.join(viz_dir, base_name + '*.png'))
                        for file in existing_files:
                            os.remove(file)
                    else:
                        name = 'scene_' + scene_name_single + '_{:03d}.png'.format(img_id)
                        norm_name = 'scene_' + scene_name_single + '_{:03d}_normal.png'.format(img_id)
                        seg_name = 'scene_' + scene_name_single + '_{:03d}_seg.png'.format(img_id)
                        # uv_name = 'scene_' + scene_name_single + '_{:03d}_uv.png'.format(img_id)
                    plt.imsave(
                        os.path.join(viz_dir, name),
                        output_viz[scene_id][img_id])
                    plt.imsave(
                        os.path.join(viz_dir, seg_name),
                        output_seg_viz[scene_id][img_id])
                    # plt.imsave(
                    #     os.path.join(viz_dir, uv_name),
                    #     output_coloruv_viz[scene_id])
                    # plt.imsave(
                    #     os.path.join(viz_dir, seg_name),
                    #     output_seg_viz[scene_id][img_id][:,:,:3])
                    if return_norm:
                        plt.imsave(
                            os.path.join(viz_dir, norm_name),
                            output_norm_viz[scene_id][img_id][:,:,:3])
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module
            code_range = cfg.get('clip_range', [-1, 1])
            # s5_time = time.time()
            # print('time_save_img:{}'.format(s5_time-s4_time))
            # decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            if self.init_code is not None:
                # decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)
                decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)
                # s6_time = time.time()
                # print(s6_time-s5_time)
        # # assert False
        return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w*6)
        # return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w//2*5)
        # return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w)

    def eval_and_viz_t(self, data, decoder, code, viz_dir=None, cfg=dict(), ortho=True, recon=False, return_norm=False):
        # s_time = time.time()
        scene_name = data['scene_name']  # (num_scenes,)
        if recon:
            test_intrinsics = data['cond_intrinsics'][:, 1::9]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['cond_poses'][:, 1::9]
            # test_intrinsics = data['cond_intrinsics'][:, 1::6]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses'][:, 1::6]
            # test_smpl_param = data['cond_smpl_param']
            # test_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses']
            test_smpl_param = data['cond_smpl_param']
        else:
            # test_intrinsics = data['test_intrinsics'][:,143:149]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['test_poses'][:,143:149]

            test_intrinsics = data['test_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['test_poses']
            test_smpl_param = data['test_smpl_param']
        num_scenes, num_imgs, _, _ = test_poses.size()
        # s2_time = time.time()
        # print(s2_time-s_time)
        if 'test_imgs' in data and not cfg.get('skip_eval', False):
            test_imgs = data['test_imgs']  # (num_scenes, num_imgs, h, w, 3)
            _, _, h, w, _ = test_imgs.size()
            test_img_paths = data['test_img_paths']  # (num_scenes, (num_imgs,))
            target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, -1, h, w)[:, :3]
            # print(target_imgs.shape)
            # target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, 4, h, w)[:, :3]
        else:
            # this one
            test_imgs = test_img_paths = target_imgs = None
            h, w = cfg['img_size']
        # image, depth = self.render(
        #     decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg)
        # image = self.render(
        #     decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg)
        if test_smpl_param.dim() == 3:
            image = []
            segs = []
            for num in range(test_smpl_param.shape[1]):
                image_single, norm_map, seg_imgs = self.render(
                    decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param[:, num], cfg=cfg, return_norm=return_norm)
                image.append(image_single)
                segs.append(seg_imgs)
            image = torch.cat(image, dim=1)
            seg_imgs = torch.cat(segs, dim=1)
        elif test_smpl_param.dim() == 2:
            # print(code.shape)
            image, norm_map, seg_imgs = self.render(
                decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg, return_norm=return_norm)
        else:
            assert False
        # pred_imgs = image.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
        # num_imgs = 50
        # pred_imgs = image.reshape(num_scenes*num_imgs, h, w, 4, 3).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 3, h, 4*w).clamp(min=0, max=1)
        pred_imgs = image.reshape(num_scenes*num_imgs, h, w, 3, 3).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 3, h, 3*w).clamp(min=0, max=1)
        # pred_imgs = image.reshape(num_scenes*num_imgs, h, w*6, 3).permute(0, 3, 1, 2).clamp(min=0, max=1)
        pred_imgs = torch.round(pred_imgs * 255) / 255

        # seg_imgs = seg_imgs.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, -1, h, w)[:, :3]
        # seg_imgs = seg_imgs.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, -1, h, w)
        seg_imgs = seg_imgs.reshape(num_scenes*num_imgs, h, w, 3, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 3*w).clamp(min=0, max=1)
        # seg_imgs = seg_imgs.reshape(num_scenes*num_imgs, h, w, 4, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 4*w).clamp(min=0, max=1)
        seg_imgs = torch.round(seg_imgs * 255) / 255
        
        if return_norm:
            # norm_map = norm_map * 0.5 + 0.5
            pred_norms = norm_map.permute(0, 1, 4, 2, 3).reshape(
            num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
            pred_norms = torch.round(pred_norms * 255) / 255
        # s3_time = time.time()
        # print(s3_time-s2_time)
        if test_imgs is not None:
            test_psnr = eval_psnr(pred_imgs[..., :1], target_imgs)
            test_ssim = eval_ssim_skimage(pred_imgs_eval, target_imgs, data_range=1)
            log_vars = dict(test_psnr=float(test_psnr.mean()),
                            test_ssim=float(test_ssim.mean()))
            if self.lpips is not None:
                if len(self.lpips) == 0:
                    lpips_eval = lpips.LPIPS(
                        net='vgg', eval_mode=True, pnet_tune=False).to(pred_imgs.device)
                    self.lpips.append(lpips_eval)
                test_lpips = []
                for pred_imgs_batch, target_imgs_batch in zip(
                        pred_imgs.split(LPIPS_BS, dim=0), target_imgs.split(LPIPS_BS, dim=0)):
                    test_lpips.append(self.lpips[0](pred_imgs_batch * 2 - 1, target_imgs_batch * 2 - 1).flatten())
                test_lpips = torch.cat(test_lpips, dim=0)
                log_vars.update(test_lpips=float(test_lpips.mean()))
            else:
                test_lpips = [math.nan for _ in range(num_scenes * num_imgs)]
        else:
            # this one
            log_vars = dict()
        # s4_time = time.time()
        # print(s4_time-s3_time)
        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)
        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            output_viz = torch.round(pred_imgs.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w*3, 3)
            output_seg_viz = torch.round(seg_imgs.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w*3)
            output_seg_viz = output_seg_viz[..., None].repeat(3, -1)
            # output_viz = torch.round(pred_imgs.permute(0, 2, 3, 1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            # output_seg_viz = torch.round(seg_imgs.permute(0, 2, 3, 1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w)
            if return_norm:
                output_norm_viz = torch.round(pred_norms.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            if test_imgs is not None:
                real_imgs_viz = (target_imgs.permute(0, 2, 3, 1) * 255).to(
                    torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 4)[..., :3]
                output_viz = np.concatenate([real_imgs_viz, output_viz], axis=-2)
            for scene_id, scene_name_single in enumerate(scene_name):
                for img_id in range(num_imgs):
                    if test_img_paths is not None:
                        base_name = 'scene_' + scene_name_single + '_' + os.path.splitext(
                            os.path.basename(test_img_paths[scene_id][img_id]))[0]
                        name = base_name + '_psnr{:02.1f}_ssim{:.2f}_lpips{:.3f}.png'.format(
                            test_psnr[scene_id * num_imgs + img_id],
                            test_ssim[scene_id * num_imgs + img_id],
                            test_lpips[scene_id * num_imgs + img_id])
                        existing_files = glob(os.path.join(viz_dir, base_name + '*.png'))
                        for file in existing_files:
                            os.remove(file)
                    else:
                        name = 'scene_' + scene_name_single + '_{:03d}.png'.format(img_id)
                        norm_name = 'scene_' + scene_name_single + '_{:03d}_normal.png'.format(img_id)
                        seg_name = 'scene_' + scene_name_single + '_{:03d}_seg.png'.format(img_id)
                    plt.imsave(
                        os.path.join(viz_dir, name),
                        output_viz[scene_id][img_id])
                    plt.imsave(
                        os.path.join(viz_dir, seg_name),
                        output_seg_viz[scene_id][img_id])
                    # plt.imsave(
                    #     os.path.join(viz_dir, seg_name),
                    #     output_seg_viz[scene_id][img_id][:,:,:3])
                    if return_norm:
                        plt.imsave(
                            os.path.join(viz_dir, norm_name),
                            output_norm_viz[scene_id][img_id][:,:,:3])
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module
            code_range = cfg.get('clip_range', [-1, 1])
            # s5_time = time.time()
            # print('time_save_img:{}'.format(s5_time-s4_time))
            # decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            if self.init_code is not None:
                # decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)
                decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)
            # s6_time = time.time()
            # print(s6_time-s5_time)
        # # assert False
        # return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w)
        # return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w*4)
        return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w*3)

    def eval_and_viz_s(self, data, decoder, code, viz_dir=None, cfg=dict(), ortho=True, recon=False, return_norm=False):
        # s_time = time.time()
        scene_name = data['scene_name']  # (num_scenes,)
        if recon:
            # test_intrinsics = data['cond_intrinsics'][:, 0::3]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses'][:, 0::3]
            # test_intrinsics = data['cond_intrinsics'][:, 1::6]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses'][:, 1::6]
            test_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['cond_poses']
            # test_smpl_param = data['cond_smpl_param']
            # test_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['cond_poses']
            test_smpl_param = data['cond_smpl_param']
        else:
            # test_intrinsics = data['test_intrinsics'][:,:6]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['test_poses'][:,:6]
            # test_intrinsics = data['test_intrinsics'][:,1::6]  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            # test_poses = data['test_poses'][:,1::6]
            test_intrinsics = data['test_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
            test_poses = data['test_poses']
            test_smpl_param = data['test_smpl_param']
        num_scenes, num_imgs, _, _ = test_poses.size()
        # if test_smpl_param.shape
        # s2_time = time.time()
        # print(s2_time-s_time)
        if 'test_imgs' in data and not cfg.get('skip_eval', False):
            test_imgs = data['test_imgs']  # (num_scenes, num_imgs, h, w, 3)
            _, _, h, w, _ = test_imgs.size()
            test_img_paths = data['test_img_paths']  # (num_scenes, (num_imgs,))
            target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, -1, h, w)[:, :3]
            # print(target_imgs.shape)
            # target_imgs = test_imgs.permute(0, 1, 4, 2, 3).reshape(num_scenes * num_imgs, 4, h, w)[:, :3]
        else:
            # this one
            test_imgs = test_img_paths = target_imgs = None
            h, w = cfg['img_size']
        # image, depth = self.render(
        #     decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg)
        # image = self.render(
        #     decoder, code, density_bitfield, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg)
        if test_smpl_param.dim() == 3:
            image = []
            seg_imgs = []
            for num in range(test_smpl_param.shape[1]):
                image_single, norm_map, seg_single = self.render(
                    decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param[:, num], cfg=cfg, return_norm=return_norm)
                image.append(image_single)
                seg_imgs.append(seg_single)
            image = torch.cat(image, dim=1)
            seg_imgs = torch.cat(seg_imgs, dim=1)
            num_imgs = test_smpl_param.shape[1]
        elif test_smpl_param.dim() == 2:
            # print(code.shape)
            image, norm_map, seg_imgs = self.render(
                decoder, code, h, w, test_intrinsics, test_poses, test_smpl_param, cfg=cfg, return_norm=return_norm)
        else:
            assert False
        # pred_imgs = image.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
        # num_imgs = 50
        pred_imgs = image.reshape(num_scenes*num_imgs, h, w, 1, 3).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 3, h, 1*w).clamp(min=0, max=1)
        # pred_imgs = image.reshape(num_scenes*num_imgs, h, w*5, 3).permute(0, 3, 1, 2).clamp(min=0, max=1)
        pred_imgs = torch.round(pred_imgs * 255) / 255

        # alpha_imgs = torch.cat([seg_imgs[..., :1], seg_imgs[..., 4:]], dim=-1)
        # # alpha_imgs = alpha_imgs.reshape(num_scenes*num_imgs, h, w*5, 1).permute(0, 3, 1, 2).clamp(min=0, max=1)
        # alpha_imgs = alpha_imgs.reshape(num_scenes*num_imgs, h, w, 1, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 1*w).clamp(min=0, max=1)
        # alpha_imgs = torch.round(alpha_imgs * 255) / 255

        # seg_imgs = seg_imgs.permute(0, 1, 4, 2, 3).reshape(
        #     num_scenes * num_imgs, -1, h, w)[:, :3]
        seg_imgs = seg_imgs.reshape(num_scenes*num_imgs, h, w, 1, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 1*w).clamp(min=0, max=1)
        # seg_imgs = seg_imgs[..., 3:].reshape(num_scenes*num_imgs, h, w, 1, 1).permute(0, 4, 1, 3, 2).reshape(num_scenes*num_imgs, 1, h, 1*w).clamp(min=0, max=1)
        seg_imgs = torch.round(seg_imgs * 255) / 255
        
        if return_norm:
            # norm_map = norm_map * 0.5 + 0.5
            pred_norms = norm_map.permute(0, 1, 4, 2, 3).reshape(
            num_scenes * num_imgs, 3, h, w).clamp(min=0, max=1)
            pred_norms = torch.round(pred_norms * 255) / 255
        # s3_time = time.time()
        # print(s3_time-s2_time)
        if test_imgs is not None:
            test_psnr = eval_psnr(pred_imgs[..., :1], target_imgs)
            test_ssim = eval_ssim_skimage(pred_imgs_eval, target_imgs, data_range=1)
            log_vars = dict(test_psnr=float(test_psnr.mean()),
                            test_ssim=float(test_ssim.mean()))
            if self.lpips is not None:
                if len(self.lpips) == 0:
                    lpips_eval = lpips.LPIPS(
                        net='vgg', eval_mode=True, pnet_tune=False).to(pred_imgs.device)
                    self.lpips.append(lpips_eval)
                test_lpips = []
                for pred_imgs_batch, target_imgs_batch in zip(
                        pred_imgs.split(LPIPS_BS, dim=0), target_imgs.split(LPIPS_BS, dim=0)):
                    test_lpips.append(self.lpips[0](pred_imgs_batch * 2 - 1, target_imgs_batch * 2 - 1).flatten())
                test_lpips = torch.cat(test_lpips, dim=0)
                log_vars.update(test_lpips=float(test_lpips.mean()))
            else:
                test_lpips = [math.nan for _ in range(num_scenes * num_imgs)]
        else:
            # this one
            log_vars = dict()
        # s4_time = time.time()
        # print(s4_time-s3_time)
        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)
        if viz_dir is not None:
            os.makedirs(viz_dir, exist_ok=True)
            output_viz = torch.round(torch.cat([pred_imgs.permute(0, 2, 3, 1), seg_imgs.permute(0, 2, 3, 1)], dim=-1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w*1, 4)
            # output_viz = output_viz.flip(2)
            output_seg_viz = torch.round(seg_imgs.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w*1)
            # output_viz = torch.round(torch.cat([pred_imgs.permute(0, 2, 3, 1), seg_imgs.permute(0, 2, 3, 1)], dim=-1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 4)
            # output_seg_viz = torch.round(seg_imgs.permute(0, 2, 3, 1) * 255).to(
            #     torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w)
            if return_norm:
                output_norm_viz = torch.round(pred_norms.permute(0, 2, 3, 1) * 255).to(
                torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 3)
            if test_imgs is not None:
                real_imgs_viz = (target_imgs.permute(0, 2, 3, 1) * 255).to(
                    torch.uint8).cpu().numpy().reshape(num_scenes, num_imgs, h, w, 4)[..., :3]
                output_viz = np.concatenate([real_imgs_viz, output_viz], axis=-2)
            for scene_id, scene_name_single in enumerate(scene_name):
                for img_id in range(num_imgs):
                    if test_img_paths is not None:
                        base_name = 'scene_' + scene_name_single + '_' + os.path.splitext(
                            os.path.basename(test_img_paths[scene_id][img_id]))[0]
                        name = base_name + '_psnr{:02.1f}_ssim{:.2f}_lpips{:.3f}.png'.format(
                            test_psnr[scene_id * num_imgs + img_id],
                            test_ssim[scene_id * num_imgs + img_id],
                            test_lpips[scene_id * num_imgs + img_id])
                        existing_files = glob(os.path.join(viz_dir, base_name + '*.png'))
                        for file in existing_files:
                            os.remove(file)
                    else:
                        name = 'scene_' + scene_name_single + '_{:03d}.png'.format(img_id)
                        norm_name = 'scene_' + scene_name_single + '_{:03d}_normal.png'.format(img_id)
                        seg_name = 'scene_' + scene_name_single + '_{:03d}_seg.png'.format(img_id)

                    plt.imsave(
                        os.path.join(viz_dir, name),
                        output_viz[scene_id][img_id])
                    plt.imsave(
                        os.path.join(viz_dir, seg_name),
                        output_seg_viz[scene_id][img_id])
                    # plt.imsave(
                    #     os.path.join(viz_dir, seg_name),
                    #     output_seg_viz[scene_id][img_id][:,:,:3])
                    if return_norm:
                        plt.imsave(
                            os.path.join(viz_dir, norm_name),
                            output_norm_viz[scene_id][img_id][:,:,:3])
            if isinstance(decoder, DistributedDataParallel):
                decoder = decoder.module
            code_range = cfg.get('clip_range', [-1, 1])
            # s5_time = time.time()
            # print('time_save_img:{}'.format(s5_time-s4_time))
            # decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            decoder.visualize(code, scene_name, viz_dir, code_range=code_range)
            if self.init_code is not None:
                # decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)
                decoder.visualize(self.init_code[None], ['000_mean'], viz_dir, code_range=code_range)
            # s6_time = time.time()
            # print(s6_time-s5_time)
        # # assert False
        return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w*1)
        # return log_vars, pred_imgs.reshape(num_scenes, num_imgs, 3, h, w)


    def mean_ema_update(self, code):
        if self.init_code is None:
            return
        mean_code = reduce_mean(code.detach().mean(dim=0))
        self.init_code.mul_(1 - self.mean_ema_momentum).add_(
            mean_code.data, alpha=self.mean_ema_momentum)

    def train_step(self, data, optimizer, running_status=None):
        raise NotImplementedError

    def val_step(self, data, viz_dir=None, show_pbar=False, **kwargs):
        decoder = self.decoder_ema if self.decoder_use_ema else self.decoder

        if 'code' in data:
            code, density_grid, density_bitfield = self.load_scene(
                data, load_density=True)
            out_rgbs = target_rgbs = None
        else:
            device = get_module_device(self)
            code = torch.load('/home/zhangweitian/HighResAvatar/cache/stage1_avatar_16bit_finalfit_sample4/code/0526.pth')
            # .to(device)
            code = code.unsqueeze(0)
            dt_gamma = 0.0
        # else:
        #     cond_imgs = data['cond_imgs']  # (num_scenes, num_imgs, h, w, 3)
        #     cond_intrinsics = data['cond_intrinsics']  # (num_scenes, num_imgs, 4), in [fx, fy, cx, cy]
        #     cond_poses = data['cond_poses']
        #     smpl_params = data['smpl_params']

        #     num_scenes, num_imgs, h, w, _ = cond_imgs.size()
        #     # (num_scenes, num_imgs, h, w, 3)
        #     # cond_rays_o, cond_rays_d = get_cam_rays(cond_poses, cond_intrinsics, h, w)
        #     # dt_gamma_scale = self.test_cfg.get('dt_gamma_scale', 0.0)
        #     # (num_scenes,)
        #     # dt_gamma = dt_gamma_scale / cond_intrinsics[..., :2].mean(dim=(-2, -1))
        #     dt_gamma = 0.0

            # with torch.enable_grad():
            #     (code, density_grid, density_bitfield,
            #      loss, loss_dict, out_rgbs, target_rgbs) = self.inverse_code(
            #         decoder, cond_imgs, cond_rays_o, cond_rays_d,
            #         dt_gamma=dt_gamma, smpl_params=smpl_params, cfg=self.test_cfg, show_pbar=show_pbar)

        # ==== evaluate reconstruction ====
        with torch.no_grad():
            if 'test_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code,
                    viz_dir=viz_dir, cfg=self.test_cfg, ortho=self.ortho, return_norm=self.return_norm)
            elif 'cond_poses' in data:
                log_vars, pred_imgs = self.eval_and_viz(
                    data, decoder, code,
                    viz_dir=viz_dir, cfg=self.test_cfg, ortho=self.ortho, recon=True, return_norm=self.return_norm)
            else:
                log_vars = dict()
                pred_imgs = None
            if out_rgbs is not None and target_rgbs is not None:
                train_psnr = eval_psnr(out_rgbs, target_rgbs)
                log_vars.update(train_psnr=float(train_psnr.mean()))
            code_rms = code.square().flatten(1).mean().sqrt()
            log_vars.update(code_rms=float(code_rms.mean()))

        # ==== save 3D code ====
        save_dir = self.test_cfg.get('save_dir', None)
        if save_dir is not None:
            self.save_scene(save_dir, code, density_grid, density_bitfield, data['scene_name'])

        # ==== outputs ====
        outputs_dict = dict(
            log_vars=log_vars,
            num_samples=len(data['scene_name']),
            pred_imgs=pred_imgs)

        return outputs_dict
