import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from pytorch3d import ops

from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES
# from pytorch3d.structures import Pointclouds
# from pytorch3d import ops
import numpy as np
import time
import cv2
import math
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_to_matrix

from .base_volume_renderer import VolumeRenderer
from ..deformers import SNARFDeformer, SMPLXDeformer, SMPLXDeformer_Abl
from ..renderers import GRenderer, GFRenderer, get_covariance, batch_rodrigues
from ..superres import SuperresolutionHybrid2X, SuperresolutionHybrid4X
from lib.ops import SHEncoder, TruncExp, all_equal


class PositionalEncoding():
    def __init__(self, input_dims=5, num_freqs=3, include_input=True):
        super(PositionalEncoding,self).__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.input_dims = input_dims
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += self.input_dims

        freq_bands = 2. ** torch.linspace(0, self.num_freqs-1, self.num_freqs)
        periodic_fns = [torch.sin, torch.cos]

        for freq in freq_bands:
            for p_fn in periodic_fns:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(math.pi * x * freq))
                # embed_fns.append(lambda x, p_fn=p_fn, freq=freq:p_fn(x * freq))
                out_dim += self.input_dims

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,coords):
        '''
        use periodic positional encoding to transform cartesian positions to higher dimension
        :param coords: [N, 3]
        :return: [N, 3*2*num_freqs], where 2 comes from that for each frequency there's a sin() and cos()
        '''
        return torch.cat([fn(coords) for fn in self.embed_fns], dim=-1)

@MODULES.register_module()
class UVPDecoder(VolumeRenderer):

    activation_dict = {
        'relu': nn.ReLU,
        'silu': nn.SiLU,
        'softplus': nn.Softplus,
        'trunc_exp': TruncExp,
        'sigmoid': nn.Sigmoid}

    def __init__(self,
                 *args,
                 interp_mode='bilinear',
                 base_layers=[3 * 32, 128],
                 density_layers=[128, 1],
                 color_layers=[128, 128, 3],
                 offset_layers=[128, 3],
                 scale_layers=[128, 3],
                 radius_layers=[128, 3],
                 cov_layers=[128, 3],
                 use_dir_enc=True,
                 dir_layers=None,
                 scene_base_size=None,
                 scene_rand_dims=(0, 1),
                 activation='silu',
                 sigma_activation='sigmoid',
                 sigmoid_saturation=0.001,
                 code_dropout=0.0,
                 flip_z=False,
                 extend_z=False,
                 gender='neutral',
                 multires=0,
                 bg_color=0,
                 image_size=1024,
                 superres=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.interp_mode = interp_mode
        self.in_chn = base_layers[0]
        self.use_dir_enc = use_dir_enc
        if scene_base_size is None:
            self.scene_base = None
        else:
            rand_size = [1 for _ in scene_base_size]
            for dim in scene_rand_dims:
                rand_size[dim] = scene_base_size[dim]
            init_base = torch.randn(rand_size).expand(scene_base_size).clone()
            self.scene_base = nn.Parameter(init_base)
        self.dir_encoder = SHEncoder() if use_dir_enc else None
        self.sigmoid_saturation = sigmoid_saturation
        self.deformer = SMPLXDeformer_Abl(gender)
        pcd_label = torch.as_tensor(np.load('work_dirs/ablation_cache/part_label_cop.npy'))
        self.register_buffer('pcd_label', pcd_label, persistent=False)
        self.renderer = GFRenderer(image_size=image_size, bg_color=bg_color, f=5000, label=self.pcd_label)
        # self.renderer = GFRenderer(image_size=image_size, bg_color=bg_color, f=5000)
        # self.renderer = GRenderer(image_size=image_size, bg_color=bg_color, f=5000)
        # self.renderer = GRenderer(image_size=image_size, bg_color=bg_color, f=2500)
        if superres:
            self.superres = None
        else:
            self.superres = None

        select_uv = torch.as_tensor(np.load('work_dirs/cache/init_uv_smplx_cop.npy'))
        self.register_buffer('select_coord', select_uv.unsqueeze(0)*2.-1., persistent=False)

        init_pcd = torch.as_tensor(np.load('work_dirs/cache/init_pcd_smplx_cop.npy'))
        self.register_buffer('init_pcd', init_pcd.unsqueeze(0), persistent=False)
        self.num_init = self.init_pcd.shape[1]

        # body_vert = torch.as_tensor(np.load('work_dirs/cache/init_vert_smplx_cop.npy'))
        # self.register_buffer('body_vert', body_vert, persistent=False)

        # body_faces = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_fa_smplx_cop.npy')).long()
        # self.register_buffer('body_faces', body_faces, persistent=False)

        # body_edge = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_edge_smplx_thu.npy')).long()
        # self.register_buffer('body_edge', body_edge, persistent=False)

        # vert_weight = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_weights_smplx_thu.npy'))
        # self.register_buffer('vert_weight', vert_weight.unsqueeze(-1).expand(-1, -1, -1, 3), persistent=False)

        # vert_idx = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_idx_smplx_thu.npy'))
        # self.register_buffer('vert_idx', vert_idx[0], persistent=False)

        # self.multires = multires
        # if multires > 0:
        #     # init_uv = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_uv_new_smplx.npy'))
        #     # self.register_buffer('init_uv', init_uv.unsqueeze(0)*2.-1.)
        #     self.pe = PositionalEncoding()
        #     uv_map = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_uvmap_smplx_cop.npy'))
        #     pcd_map = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/init_posmap_smplx_cop.npy'))
        #     input_coord = torch.cat([pcd_map, uv_map], dim=1)
        #     # self.register_buffer('input_freq', input_coord, persistent=False)
        #     pos_embed = self.pe.embed(input_coord.reshape(-1, 256*256).permute(1, 0))
        #     pos_embed = pos_embed.permute(1, 0).reshape(-1, 256, 256).unsqueeze(0)
        #     self.register_buffer('input_freq', pos_embed, persistent=False)
            # input_freq = self.pe.embed(input_coord)
            # self.register_buffer('input_freq', input_freq.unsqueeze(0), persistent=False)

            # base_layers[0] += self.pe.out_dim
            # color_layers[0] += self.pe.out_dim
        #     base_layers[0] += 35
        #     color_layers[0] += 35
        # else:
        self.init_uv = None

        activation_layer = self.activation_dict[activation.lower()]


        base_net = [] # linear (in=18, out=64, bias=True)
        for i in range(len(base_layers) - 1):
            # base_net.append(nn.Linear(base_layers[i], base_layers[i + 1]))
            base_net.append(nn.Conv2d(base_layers[i], base_layers[i + 1], 7, padding=3))
            if i != len(base_layers) - 2:
                base_net.append(nn.BatchNorm2d(base_layers[i+1]))
                base_net.append(activation_layer())
        self.base_net = nn.Sequential(*base_net)
        self.base_bn = nn.BatchNorm2d(base_layers[-1])
        self.base_activation = activation_layer()

        density_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(density_layers) - 1):
            # density_net.append(nn.Linear(density_layers[i], density_layers[i + 1]))
            density_net.append(nn.Conv2d(density_layers[i], density_layers[i + 1], 1))
            if i != len(density_layers) - 2:
                density_net.append(nn.BatchNorm2d(density_layers[i+1]))
                density_net.append(activation_layer())
        density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.density_net = nn.Sequential(*density_net)

        offset_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(offset_layers) - 1):
            # density_net.append(nn.Linear(density_layers[i], density_layers[i + 1]))
            offset_net.append(nn.Conv2d(offset_layers[i], offset_layers[i + 1], 1))
            if i != len(offset_layers) - 2:
                offset_net.append(nn.BatchNorm2d(offset_layers[i+1]))
                offset_net.append(activation_layer())
        # density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.offset_net = nn.Sequential(*offset_net)

        self.dir_net = None
        tex_net = [] # linear(in=64, out=3, bias=True), sigmoid
        color_net = []
        cov_net = []
        for i in range(len(color_layers) - 2):
            # color_net.append(nn.Linear(color_layers[i], color_layers[i + 1]))
            tex_net.append(nn.Conv2d(color_layers[i], color_layers[i + 1], kernel_size=7, padding=3))
            tex_net.append(nn.BatchNorm2d(color_layers[i+1]))
            tex_net.append(activation_layer())
        color_net.append(nn.Conv2d(color_layers[-2], color_layers[-1], kernel_size=1))
        color_net.append(nn.Sigmoid())
        cov_net.append(nn.Conv2d(cov_layers[-2], cov_layers[-1], kernel_size=1))
        cov_net.append(nn.Sigmoid())
        self.color_net = nn.Sequential(*color_net)
        self.cov_net = nn.Sequential(*cov_net)
        self.tex_net = nn.Sequential(*tex_net)
        self.code_dropout = nn.Dropout2d(code_dropout) if code_dropout > 0 else None

        self.flip_z = flip_z
        self.extend_z = extend_z

        init_rot = torch.as_tensor(np.load('work_dirs/cache/init_rot_smplx_cop.npy'))
        self.register_buffer('init_rot', init_rot, persistent=False)

        face_mask = torch.as_tensor(np.load('work_dirs/cache/face_mask_cop.npy'))
        self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

        hands_mask = torch.as_tensor(np.load('work_dirs/cache/hands_mask_cop.npy'))
        self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

        outside_mask = torch.as_tensor(np.load('work_dirs/cache/outside_mask_cop.npy'))
        self.register_buffer('outside_mask', outside_mask.unsqueeze(0), persistent=False)

        # mouth_in_mask = torch.as_tensor(np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/mouth_in_mask_thu.npy'))
        # self.register_buffer('mouth_in_mask', mouth_in_mask.unsqueeze(0), persistent=False)

        nose_mask = torch.as_tensor(np.load('work_dirs/cache/nose_mask_thu.npy'))
        self.register_buffer('nose_mask', nose_mask.unsqueeze(0), persistent=False)

        # cloth_mask = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/cloth_temp_mask_thu.npy'))
        # self.register_buffer('cloth_mask', cloth_mask, persistent=False)

        # hair_mask = torch.as_tensor(np.load('/mnt/sdb/zwt/LayerAvatar/work_dirs/cache/hair_mask_thu.npy'))
        # self.register_buffer('hair_mask', hair_mask, persistent=False)
        all_mask = torch.as_tensor(np.load('work_dirs/ablation_cache/all_mask.npy'))
        self.register_buffer('all_mask', all_mask, persistent=False)

        offset_mask = (self.face_mask + self.outside_mask) * (~self.nose_mask)
        # * (~self.nose_mask)
        self.register_buffer('offset_mask', offset_mask, persistent=False)

        mask_other = self.pcd_label != 1
        self.register_buffer('mask_other', mask_other, persistent=False)

        attr_weight = torch.tensor([2, 3, 3, 3, 1, 1, 1, 1, 1, 1, 3, 3, 3])
        self.register_buffer('attr_weight', attr_weight.reshape(1, -1, 1, 1), persistent=False)
        # num_body, num_cloth, num_hair = init_pcd.shape[0], cloth_mask.sum(), hair_mask.sum()
        # pcd_label = torch.zeros(num_body*3, 3)
        # pcd_label[:num_body, 0] = 1
        # pcd_label[num_body:num_body*2, 1] = cloth_mask
        # pcd_label[-num_body:, 2] = hair_mask

        # face_vert_mask = self.face_mask[:, self.vert_idx].sum(-1).bool()
        # self.register_buffer('face_vert_mask', face_vert_mask , persistent=False)

        # rm_mask = self.outside_mask * (~self.nose_mask)
        # self.register_buffer('rm_mask', rm_mask, persistent=False)
        # lip_mask = torch.as_tensor(np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/nose_lip_thu.npy'))
        # self.register_buffer('lip_mask', lip_mask.unsqueeze(0), persistent=False)

        # feet_mask = torch.as_tensor(np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/feet_mask_thu.npy'))
        # self.register_buffer('feet_mask', feet_mask.unsqueeze(0), persistent=False)

        # ear_mask = torch.as_tensor(np.load('/home/zhangweitian/HighResAvatar/work_dirs/cache/ear_mask_thu.npy'))
        # self.register_buffer('ear_mask', ear_mask.unsqueeze(0), persistent=False)

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net[-1], 0)
        if self.offset_net is not None:
            self.offset_net[-1].weight.data.uniform_(-1e-5, 1e-5)
            self.offset_net[-1].bias.data.zero_()
        if self.cov_net is not None:
            self.cov_net[0].weight.data.uniform_(-1e-5, 1e-5)
            self.cov_net[0].bias.data.zero_()
        # nn.init.constant_(self.color_net[-2].weight, 0)
        # nn.init.constant_(self.color_net[-2].bias, 0)
        # nn.init.constant_(self.scale_net[0].weight, 0)
        # nn.init.constant_(self.scale_net[0].bias, 0)
        # nn.init.constant_(self.radius_net[0].weight, 0)
        # nn.init.constant_(self.radius_net[0].bias, 0)
        # if self.density_net is not None:
        #     # nn.init.constant_(self.density_net[0].bias, 2.1972) # 0.9
        #     nn.init.constant_(self.density_net[0].bias, -1.3863) # 0.2
            # nn.init.constant_(self.density_net[0].bias, 0)

    def xyz_transform(self, xyz, smpl_params=None, num_scenes=1):
        if xyz.dim() != 3:
            print(xyz.shape)
            assert False
        self.deformer.prepare_deformer(smpl_params, num_scenes, device=xyz.device)
        xyz, tfs = self.deformer(xyz, mask=(self.face_mask+self.hands_mask+self.outside_mask), cano=False)
        # xyz, tfs = self.deformer(xyz, mask=None, cano=False)
        return xyz, tfs

    def extract_pcd(self, code, smpl_params, init=False):
        if isinstance(code, list):
            num_scenes, _, h, w = code[0].size()
        elif code.dim() == 5:
            num_scenes, n_layer, n_channels, h, w = code.size()
        else:
            num_scenes, n_channels, h, w = code.size()
        init_pcd = self.init_pcd.repeat(num_scenes, 5, 1)
        # init_vert = self.body_vert.expand(num_scenes, -1, -1)
        # face_vert_mask = self.face_vert_mask.expand(num_scenes, -1)
        
        if self.code_dropout is not None:
            # code = self.code_dropout(
            #     code.reshape(num_scenes, n_channels, h, w)
            # ).reshape(num_scenes, n_channels, h, w)
            assert False
        if self.scene_base is not None:
            code = code + self.scene_base
            assert False
        
        sigmas, rgbs, radius, rot, offset, attrmap = self._decode(code, init=init)
        # normals = (self.init_rot[:, :, -1]).unsqueeze(0).expand(num_scenes, -1, -1)
        # normals = (self.init_rot[:, :, -1]).reshape(num_scenes, -1, 3)
        # vert_body = (offset[:, self.vert_idx, 0] * self.vert_weight).sum(2).clamp(-1, 1) + init_vert
        # body_pcd = (vert_body[:, self.body_faces[:, 0]] + vert_body[:, self.body_faces[:, 1]] + vert_body[:, self.body_faces[:, 2]]) / 3
        # edge_ori = (init_vert[:, self.body_edge[:, 0]] - init_vert[:, self.body_edge[:, 1]])
        # edge_tran = (vert_body[:, self.body_edge[:, 0]] - vert_body[:, self.body_edge[:, 1]])
        # edge_off = ((edge_tran - edge_ori) ** 2).mean()
        hands_color = rgbs[:, :self.hands_mask.shape[1]][:, self.hands_mask[0]]
        mean_body_color = hands_color.mean(1)

        canon_pcd = init_pcd + offset

        # mean_color = (rgbs[:, self.vert_idx] * self.vert_weight).sum(2)
        # reg_mean_color = rgbs[:, self.vert_idx] - mean_color.unsqueeze(2)
        # reg_body_color = reg_mean_color[~face_vert_mask].abs().mean()
        # R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
        # body_pcd = init_pcd + offset[:, :, 0]
        # cloth_pcd = body_pcd + ((offset[:, :, 1]).clamp(0)+1e-2) * normals
        # hair_pcd = body_pcd + ((offset[:, :, 2]).clamp(0)+1e-2) * normals
        # cloth_pcd = body_pcd + offset[:, :, 1] * normals
        # hair_pcd = body_pcd + offset[:, :, 2] * normals
        # canon_pcd = torch.cat([body_pcd, cloth_pcd, hair_pcd], dim=1)
        defm_pcd, tfs = self.xyz_transform(canon_pcd, smpl_params, num_scenes)
        
        return defm_pcd, sigmas, rgbs, offset, radius, tfs, rot, mean_body_color, attrmap
    
    def _decode(self, point_code, init=False):
        if isinstance(point_code, list):
            assert False
            num_scenes, _, h, w = point_code[0].shape
            geo_code, tex_code = point_code
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            # init_coord = self.init_pcd.repeat(num_scenes, 1, 1).permute(0, 2, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1, 1)
        elif point_code.dim() == 4:
            num_scenes, n_channels, h, w = point_code.shape
            # point_code = point_code.reshape(num_scenes*3, n_channels//3, h, w)
            select_coord = self.select_coord.unsqueeze(1).expand(num_scenes, -1, -1, -1)
            # select_coord = select_coord[:, self.pcd_label.sum(-1).bool()]
            # init_coord = self.init_pcd.repeat(num_scenes, 1, 1).permute(0, 2, 1)
            # point_code = point_code.reshape(num_scenes*3, -1, h, w)
            # if self.multires != 0:
                # input_freq = self.input_freq.repeat(num_scenes*3, 1, 1, 1)
                # input_code = torch.cat([point_code, input_freq], dim=1)
            # body_code, cloth_code, hair_code = point_code.split(12, dim=1)
            # uv_coord = self.init_uv if self.init_uv is None else self.init_uv.repeat(num_scenes, 1, 1)
        elif point_code.dim() == 5:
            num_scenes, n_layer, n_channels, h, w = point_code.shape
            point_code = point_code.reshape(num_scenes*n_layer, n_channels, h, w)
            select_coord = self.select_coord.unsqueeze(1).expand(num_scenes*n_layer, -1, -1, -1)
            if self.multires != 0:
                input_freq = self.input_freq.expand(num_scenes*n_layer, -1, -1, -1)
        else:
            assert False

        geo_code, tex_code = point_code.split(6, dim=1)
        base_x = self.base_net(geo_code)
        base_x_act = self.base_activation(self.base_bn(base_x))
        sigma = self.density_net(base_x_act)
        offset = self.offset_net(base_x_act)
        base_tex = self.tex_net(tex_code)
        rgbs = self.color_net(base_tex)
        radius_rot = self.cov_net(base_tex)
        outputs = torch.cat([sigma, offset, rgbs, radius_rot], dim=1) # [1, 13*5, 256, 256]
        # print(rgbs_radius_rot.shape)
        # assert False
        # rgbs, radius, rot = rgbs_radius_rot.split([9, 9, 9], dim=1)
        # rgbs = rgbs.reshape(1, 3, 3, 256, 256)
        # radius = radius.reshape(1, 3, 3, 256, 256)
        # rot = rot.reshape(1, 3, 3, 256, 256)
        # offset_min = offset.reshape(1, 3, 3, -1).min(3)[0]
        # offset_max = offset.reshape(1, 3, 3, -1).max(3)[0]
        # offsets = (offset.reshape(1, 3, 3, 256, 256) - offset_min.unsqueeze(3).unsqueeze(4)) / (offset_max - offset_min).unsqueeze(3).unsqueeze(4)
        # sigma = sigma.reshape(1, 3, 1, 256, 256)
        # for i in range(3):
        #     cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/viz_ad_32/offsets_{}.png'.format(i), (offsets[0, i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
        #     cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/viz_ad_32/rgbs_{}.png'.format(i), (rgbs[0, i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
        #     cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/viz_ad_32/radius_{}.png'.format(i), (radius[0, i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
        #     cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/viz_ad_32/rot_{}.png'.format(i), (rot[0, i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
        #     cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/viz_ad_32/sigma_{}.png'.format(i), (sigma[0, i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
        # assert False
        # rgbs, radius, rot = rgbs_radius_rot.split([3, 3, 3], dim=1)
        # for i in range(8):
        #     offset_min = offset.reshape(8, 3, -1).min(2)[0]
        #     offset_max = offset.reshape(8, 3, -1).max(2)[0]
        #     print(offset_min.shape)
        #     print(offset.shape)
        #     offsets = (offset - offset_min.unsqueeze(2).unsqueeze(3)) / (offset_max - offset_min).unsqueeze(2).unsqueeze(3)
        #     cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/offset_{}.png'.format(i), (offsets[i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
            # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/sigma_{}.png'.format(i), (sigma[i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
            # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/rgb_{}.png'.format(i), (rgbs[i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
            # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/radius_{}.png'.format(i), (radius[i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
            # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/rot_{}.png'.format(i), (rot[i].permute(1, 2, 0)).cpu().numpy()[:,:,::-1]*255)
        # assert False
        # outputs = torch.cat([sigma, offset, rgbs_radius_rot], dim=1)
        # rgbs_png, radius_png, rot_png = rgbs_radius_rot.split([9, 9, 9], dim=1)
        # offset_png = offset.reshape(num_scenes, -1, 3, 256, 256)
        # rgbs_png = rgbs_png.reshape(num_scenes, 3, 3, 256, 256)
        # radius_png = radius_png.reshape(num_scenes, -1, 3, 256, 256)
        # rot_png = rot_png.reshape(num_scenes, -1, 3, 256, 256)
        # for i in range(2):
            # for j in range(3):
                # cv2.imwrite(f'/mnt/sdb/zwt/LayerAvatar/debug/viz/sigma_{i}_{j}.png', (sigma_png[i, :, j].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
                # cv2.imwrite(f'/mnt/sdb/zwt/LayerAvatar/debug/viz/offset_{i}_{j}.png', (offset_png[i, :, j].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
                # cv2.imwrite(f'/mnt/sdb/zwt/LayerAvatar/debug/viz/radius_{i}_{j}.png', (radius_png[i, :, j].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
                # cv2.imwrite(f'/mnt/sdb/zwt/LayerAvatar/debug/viz/rot_{i}_{j}.png', (rot_png[i, :, j].permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
        # coord_in = geo_code if self.init_uv is None else torch.cat([geo_code, coord, uv_coord], dim=-1)
        output_attr = F.grid_sample(outputs, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, 13*5, -1).permute(0, 2, 1)
        # 4, 50200, 65
        # output_attr = F.grid_sample(outputs, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False)
        # output_attr = output_attr.reshape(num_scenes, 3, 13, -1).permute(0, 2, 1, 3).reshape(num_scenes, 3*13, -1).permute(0, 2, 1)
        # output_attr = F.grid_sample(outputs, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, 39, -1).permute(0, 2, 1)
        # sigma, offset, rgbs, radius, rot = output_attr.split([1, 3, 3, 3, 3], dim=2)
        sigma, offset, rgbs, radius, rot = output_attr.split([5, 15, 15, 15, 15], dim=2)
        offset = offset.reshape(num_scenes, -1, 5, 3).permute(0, 2, 1, 3).flatten(1, 2)
        rot = rot.reshape(num_scenes, -1, 5, 3).permute(0, 2, 1, 3).flatten(1, 2)
        radius = radius.reshape(num_scenes, -1, 5, 3).permute(0, 2, 1, 3).flatten(1, 2)
        rgbs = rgbs.reshape(num_scenes, -1, 5, 3).permute(0, 2, 1, 3).flatten(1, 2)
        sigma = sigma.reshape(num_scenes, -1, 5, 1).permute(0, 2, 1, 3).flatten(1, 2)
        # print(sigma.shape)
        # print(self.mouth_in_mask.shape)
        # assert False
        # rgbs[:, self.lip_mask[0]] = torch.tensor([0., 0., 1.]).to(offset.device)
        # sigma[self.mouth_in_mask.expand(num_scenes, -1)] = 0
        # sigma = sigma * 0 + 1
        # offset *= 0
        # sigma[self.rm_mask.expand(num_scenes, -1)] = 0.0
        # sigma[self.hands_mask.expand(num_scenes, -1)] = 1.0
        # sigma[self.nose_mask.expand(num_scenes, -1)] = 1.0
        # offset[self.nose_mask.expand(num_scenes, -1)] = 0.0
        # offset[self.hands_mask.expand(num_scenes, -1)] = 0.0
        # offset[self.hands_mask.expand(num_scenes, -1)] = offset[self.hands_mask.expand(num_scenes, -1)].clip(0, 0.005)
        # radius[self.hands_mask.expand(num_scenes, -1)] = radius[self.hands_mask.expand(num_scenes, -1)].clip(0, 0.5)

        # select_coord = select_coord.permute(0, 1, 3, 2)
        # geo_code, tex_code = point_code.reshape(num_scenes, n_channels, -1).split(16, dim=1)
        # geo_code, tex_code = code_in.reshape(num_scenes, n_channels, -1).split(16, dim=1)
        # tex_code = torch.flip(tex_code, [0])
        # base_in = geo_code if self.multires == 0 else torch.cat([geo_code, init_coord, select_coord[:, 0]], dim=1)

        # color_in = torch.cat([base_x, tex_code], dim=1) if self.multires == 0 else torch.cat([base_x, tex_code, init_coord, select_coord[:, 0]], dim=1)
        # color_in = tex_code if self.multires == 0 else torch.cat([tex_code, init_coord, select_coord[:, 0]], dim=1)
        # color_in = tex_code if self.multires == 0 else torch.cat([tex_code, input_freq], dim=1)
        # radius = self.scale_net(color_in).permute(0, 2, 1)
        # rot = self.radius_net(color_in).permute(0, 2, 1)
        # rgbs_radius_rot = self.color_net(color_in).permute(0, 2, 1)
        # rgbs, radius, rot = rgbs_radius_rot.split([3, 3, 3], dim=2)
        # offset, sigma = offset_sigmas.split([3, 1], dim=1)

        # if init:
        #     sigmas = sigmas * 0 + 0.9

        # if self.use_dir_enc:
        #     dirs = torch.cat(dirs, dim=0) if num_scenes > 1 else dirs[0]
        #     sh_enc = self.dir_encoder(dirs)
        #     if self.dir_net is not None:
        #         color_in = self.base_activation(base_x + self.dir_net(sh_enc))
        #     else:
        #         color_in = torch.cat([base_x_feat, sh_enc], dim=-1)
        # else:
        #     # color_in = base_x_act
        # color_in = torch.cat([base_x_act, tex_code, coord, uv_coord])
        # color_in = torch.cat([base_x_act, tex_code], dim=1)

        # output = torch.cat([sigmas, offset, rgbs_radius_rot], dim=1).view(num_scenes, 13, -1).permute(0, 2, 1)
        # outputs = F.grid_sample(output, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, 13, -1).permute(0, 2, 1)
        # radius, rot = radius_rot.split([3, 3], dim=2)
        # sigma = self.sig_density(sigma)
        # rgbs = self.sig_color(rgbs)
        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation
        # rgb_uvmap = rgbs.reshape(num_scenes, 256, 256, 3)
        # rgb_uvmap = torch.round(rgb_uvmap*255).cpu().numpy()[:,:,:,::-1]
        # import cv2 
        # for i in range(num_scenes):
        #     cv2.imwrite(f'/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/{i}.png', rgb_uvmap[i])
        # assert False
        # sigma = sigma * 0 + 1
        # sigma[self.offset_mask.expand(num_scenes, -1)] = sigma[self.offset_mask.expand(num_scenes, -1)] * 0
        # rgbs = rgbs * 0 + 0.5
        # radius = radius.clip(-1, 1)
        radius = (radius - 0.5) * 2
        rot = (rot - 0.5) * np.pi
        # rot *= 0
        # rot[self.nose_mask.expand(num_scenes, -1)] = rot[self.nose_mask.expand(num_scenes, -1)] * 0.
        # rot = rot.clip(-np.pi/2, np.pi/2)
        # outputs [4, 65, 128, 128]
        return sigma, rgbs, radius, rot, offset, outputs

    def gaussian_render(self, pcd, sigmas, rgbs, normals, cov3D, num_scenes, num_imgs, cameras, use_scale=False, radius=None, return_norm=False, return_viz=False, mask=None):
    # def gaussian_render(self, pcd, sigmas, rgbs, num_scenes, num_imgs, num_points, cameras):
        #TODO: add mask or visible points to images or select ind to images
        # mask = sigmas > 0.5
        # s_time = time.time()
        assert num_scenes == 1
        
        # pcd = pcd.reshape(-1, 3)
        # pcd = pcd.reshape(num_scenes, 3, -1, 3)
        if use_scale:
            # pcd = pcd.reshape(num_scenes*3, -1, 3)
            pcd = pcd.reshape(5, -1, 3)
            # pcd = pcd.reshape(num_scenes, 3, -1, 3)
            # dist2 = torch.clamp_min(distCUDA2(self.init_pcd[0]), 0.0000001)
            dist2 = torch.cat([torch.clamp_min(distCUDA2(pt), 0.0000001) for pt in pcd], dim=0)
            # dist2 = torch.clamp_min(distCUDA2(pcd[0, 0]), 0.0000001)
            # scales = (0.1 * torch.sqrt(dist2)[...,None].repeat(1, 3)).detach()
            scales = torch.sqrt(dist2)[...,None].repeat(1, 3).detach()
            scales[:, -1] *= 0.1
            scale = (radius+1)*scales
            cov3D = get_covariance(scale, cov3D).reshape(-1, 6)
        pcd = pcd.reshape(-1, 3)
        # pcd_label = torch.ones_like(pcd)
        # label0, label1, label2 = pcd_label.chunk(3)
        # pcd_label = torch.cat([label0, label1+1, label2+2], dim=0) / 255
        # sigmas[:50200] *= 0
        # sigmas[-50200:] *= 0
        # s1_time = time.time()
        # print(s1_time - s_time)
        # feat = torch.cat([rgbs, normals], dim=-1)
        images_all = []
        segs_all = []
        viz_masks = [] if return_viz else None
        norm_all = [] if return_norm else None

        # R_delta = batch_rodrigues(rot)
        # R = torch.bmm(self.init_rot[masks], R_delta)
        # R_def = torch.bmm(tfs[:,:3,:3], R)
        # scale = (radius+1)*self.scales[masks]
        # cov3D = get_covariance(scale, R_def)
        # if mask != None:
        pcd = pcd[self.all_mask]
        rgbs = rgbs[self.all_mask]
        sigmas = sigmas[self.all_mask]
        cov3D = cov3D[self.all_mask]
        # normals = normals[self.all_mask]
        if return_norm:
            assert False
            for i in range(num_imgs):
                self.renderer.prepare(cameras[i])
                # w2c = cameras[i][3:].reshape(4, 4)
                # normals = einsum('pij,pj->pi', w2c[:3,:3].unsqueeze(0).expand(normals.shape[0], -1, -1), normals)
                image = self.renderer.render_gaussian(means3D=pcd, colors_precomp=rgbs, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                images_all.append(image)
                norm = self.renderer.render_gaussian(means3D=pcd, colors_precomp=normals, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                norm_all.append(norm)
            norm_all = torch.stack(norm_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        else:
            for i in range(num_imgs):
                self.renderer.prepare(cameras[i])
                # w2c_ = cameras[i][3:].clone().reshape(4, 4)[:3, :3].unsqueeze(0)
                # w2c_[:, 1:,:] *= -1
                # normals_trans = torch.bmm(w2c_.expand(normals.shape[0], -1, -1), normals.unsqueeze(-1))
                # normals_vis = normals_trans * 0.5 + 0.5
                # normals_vis = normals_vis[..., 0]
                # normals_vis = normals * 0.5 + 0.5
                # image = self.renderer.render_gaussian(means3D=pcd, colors_precomp=normals_vis, 
                #     rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                image, seg = self.renderer.render_gaussian(means3D=pcd, colors_precomp=rgbs, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D, label=self.pcd_label)
                images_all.append(image)
                # seg = self.renderer.render_gaussian(means3D=pcd, colors_precomp=normals, 
                #     rotations=None, opacities=sigmas*0+1, scales=None, cov3D_precomp=cov3D)
                segs_all.append(seg)
                if return_viz:
                    assert False
                    viz_mask = self.renderer.render_gaussian(means3D=pcd, colors_precomp=pcd.clone(), 
                        rotations=None, opacities=sigmas*0+1, scales=None, cov3D_precomp=cov3D)
                    viz_masks.append(viz_mask)
        images_all = torch.stack(images_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        segs_all = torch.stack(segs_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        # s2_time = time.time()
        # print(s2_time-s1_time)
        if return_viz:
            assert False
            viz_masks = torch.stack(viz_masks, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 3)
            dist_sq, idx, neighbors = ops.knn_points(pcd.unsqueeze(0), viz_masks[:, ::10], K=1, return_nn=True)
            # viz_masks = (dist_sq < 0.0001)[0]
            viz_masks = (dist_sq < 0.0005)[0]

        # s3_time = time.time()
        # print(s3_time-s2_time)
        if use_scale:
            return images_all, norm_all, viz_masks, scale, segs_all
        else:
            return images_all, norm_all, viz_masks, segs_all

    # def point_density_decode(self, xyzs, code, smpl_params=None, **kwargs):
    #     sigmas, _, offset, radius, num_points = self.point_decode(
    #         xyzs, None, code, smpl_params=smpl_params, density_only=True, **kwargs)
    #     return sigmas, offset, radius, num_points

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, num_chn, h, w = code.size()
        code_viz = code.reshape(num_scenes, 3, 4, h, w).cpu().numpy()
        if not self.flip_z:
            code_viz = code_viz[..., ::-1, :]
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, 4 * w)
        for code_single, code_viz_single, scene_name_single in zip(code, code_viz, scene_name):
            torch.save(code_single, os.path.join(viz_dir, 'scene_' + scene_name_single + '.pth'))
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])

    def forward(self, code, grid_size, smpl_params, cameras, num_imgs,
                dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False, return_norm=False, init=False, mask=None):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        if isinstance(code, list):
            num_scenes = len(code[0])
        else:
            num_scenes = len(code)
        assert num_scenes > 0

        code_h, code_w = code.shape[-2:]

        if self.training:
            image = []
            segs = []
            scales = []
            norm = [] if return_norm else None
            # xyzs, sigmas, rgbs, offsets, radius, tfs, rot, edge_off, reg_color = self.extract_pcd(code, smpl_params, init=init)
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot, reg_color, attrmap = self.extract_pcd(code, smpl_params, init=init)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes*5, 1, 1), R_delta)
            # R = self.init_rot.repeat(num_scenes, 1, 1)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            # R_norm = torch.bmm(tfs.flatten(0, 1).permute(0, 2, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            # for camera_single, cov3D_single, pcd_single, rgbs_single, sigmas_single, normal_single in zip(cameras, cov3D, xyzs, rgbs, sigmas, normals):
                # image_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, cov3D_single, 1, num_imgs, camera_single)
            if return_norm:
                assert False
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, norm_single, _, _, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, return_norm=True)
                    image.append(image_single)
                    norm.append(norm_single)
                norm = torch.cat(norm, dim=0)
            else:
                # rgbs, sigmas, radius
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, _, _, scale, seg_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single)
                    image.append(image_single)
                    segs.append(seg_single)
                    scales.append(scale.unsqueeze(0))
            # assert False
            #     offset.append(offset_single)
            #     part_mask.append(self.part_mask[mask_single])
            #     scales.append(scale_single)
            
            image = torch.cat(image, dim=0)
            segs = torch.cat(segs, dim=0)
            scales = torch.cat(scales, dim=0)
            # skin_color = rgbs[:, :50200][self.hands_mask.expand(num_scenes, -1)].mean(0).detach()
            # offset = torch.cat(offset, dim=0)
            # scales = torch.cat(scales, dim=0)
            # part_mask = torch.cat(part_mask, dim=0)
            offset_dist = (offsets ** 2).sum(-1)[:, self.all_mask]
            weighted_offset = torch.mean(offset_dist[:, ~self.mask_other]) + torch.mean(offset_dist[:, self.mask_other]) * 0.1 + torch.mean(offset_dist[:, :self.hands_mask.shape[1]][self.hands_mask.expand(num_scenes, -1)]) + torch.mean(offset_dist[:, :self.offset_mask.shape[1]][self.offset_mask.expand(num_scenes, -1)])
            # weighted_offset = torch.mean(offset_dist) + torch.mean(offset_dist[self.hands_mask.repeat(num_scenes, 1)]) + torch.mean(offset_dist[self.offset_mask.repeat(num_scenes, 1)]) + torch.mean(offset_dist[:50200])*10

            # attrmap = attrmap.reshape(num_scenes, -1, code_h, 3, code_w//3).permute(0, 3, 1, 2, 4).flatten(0, 1)
            # print(self.attr_weight.repeat(num_scenes, 1, 5, 1).reshape(1, 65, 1, 1))
            attrmap = attrmap * self.attr_weight.repeat(num_scenes, 1, 5, 1).reshape(num_scenes, 65, 1, 1)
            # print(attrmap.shape)
            # assert False
            results = dict(
                # part_mask=part_mask,
                reg_color=reg_color,
                # edge=edge_off,
                # skin_color=skin_color,
                sigmas=sigmas,
                segs=segs,
                scales=scales,
                attrmap=attrmap,
                # norm=norm,
                image=image,
                offset=weighted_offset)
        else:
            if isinstance(code, list):
                device = code[0].device
            else:
                device = code.device
            dtype = torch.float32

            image = []
            segs = []
            offsets = None
            scale = None
            # part_mask = None
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot, _, _ = self.extract_pcd(code, smpl_params, init=False)
            # longer nose
            # z_min = xyzs[:, self.nose_mask[0], :,-1].min(1)[0]
            # z_min = z_min.unsqueeze(1)
            # xyzs[:, self.nose_mask[0], :,-1] *= 2
            # xyzs[:, self.nose_mask[0], :,-1] -= z_min
            # longer ears
            # y_max = xyzs[:, self.ear_mask[0], :,1].max(1)[0]
            # y_max = y_max.unsqueeze(1)
            # xyzs[:, self.ear_mask[0], :,1] *= 1.5 
            # xyzs[:, self.ear_mask[0], :,1] -= (y_max * 0.5)
            # transfer
            # mask_transfer = self.face_mask + self.nose_mask
            # tfs[:, mask_transfer[0]] = torch.flip(tfs[:, mask_transfer[0]], dims=[0])
            # xyzs[:, mask_transfer[0]] = torch.flip(xyzs[:, mask_transfer[0]], dims=[0])
            # offsets[:, mask_transfer[0]] = torch.flip(offsets[:, mask_transfer[0]], dims=[0])
            # rgbs[:, mask_transfer[0]] = torch.flip(rgbs[:, mask_transfer[0]], dims=[0])
            # radius[:, mask_transfer[0]] = torch.flip(radius[:, mask_transfer[0]], dims=[0])
            # sigmas[:, mask_transfer[0]] = torch.flip(sigmas[:, mask_transfer[0]], dims=[0])
            # rot[:, mask_transfer[0]] = torch.flip(rot[:, mask_transfer[0]], dims=[0])

            # tfs[:, self.feet_mask[0]] = torch.flip(tfs[:, self.feet_mask[0]], dims=[0])
            # xyzs[:, self.feet_mask[0]] = torch.flip(xyzs[:, self.feet_mask[0]], dims=[0])
            # offsets[:, self.feet_mask[0]] = torch.flip(offsets[:, self.feet_mask[0]], dims=[0])
            # rgbs[:, self.feet_mask[0]] = torch.flip(rgbs[:, self.feet_mask[0]], dims=[0])
            # radius[:, self.feet_mask[0]] = torch.flip(radius[:, self.feet_mask[0]], dims=[0])
            # sigmas[:, self.feet_mask[0]] = torch.flip(sigmas[:, self.feet_mask[0]], dims=[0])
            # rot[:, self.feet_mask[0]] = torch.flip(rot[:, self.feet_mask[0]], dims=[0])
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes*5, 1, 1), R_delta)
            # R = self.init_rot.repeat(num_scenes, 1, 1)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            # normals = einsum('pij,pj->pi', tfs.flatten(0, 1)[:,:3,:3].permute(0, 2, 1), R[:,:,-1])
            # normals = (R_def[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
            # normals = (R_def[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            # normals = (R_norm[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            # print(tfs.flatten(0, 1)[12, :3, :3])
            # print(tfs.flatten(0, 1).inverse().transpose(-2,-1)[12, :3, :3])
            # assert False
            # scale = (radius.reshape(-1, 3)+1)*self.scales.repeat(num_scenes, 1)
            # cov3D = get_covariance(scale, R_def).reshape(num_scenes, -1, 6)
            # normals = (R_def[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            # for camera_single, cov3D_single, pcd_single, rgbs_single, sigmas_single, normal_single in zip(cameras, cov3D, xyzs, rgbs, sigmas, normals):
            if mask != None:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single, mask_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius, mask):
                    image_single, _, viz_masks, _, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, mask=mask_single)
                    image.append(image_single)
            else:
                viz_masks = []
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, _, viz_mask, _, seg_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, return_viz=False)
                    image.append(image_single)
                    segs.append(seg_single)
                    viz_masks.append(viz_mask)
                # viz_masks = torch.cat(viz_masks, dim=0)
                viz_masks = None
                # print(image_single.shape)
            image = torch.cat(image, dim=0)
            segs = torch.cat(segs, dim=0)

            results = dict(
                # part_mask=part_mask,
                # scales=scale.reshape(num_scenes, -1, 3),
                segs=segs,
                viz_masks=viz_masks,
                scales=None,
                image=image,
                offset=offsets)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results