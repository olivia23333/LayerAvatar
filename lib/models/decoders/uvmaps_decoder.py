import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES
# from pytorch3d.structures import Pointclouds
# from pytorch3d import ops
import numpy as np
import math
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_to_matrix

from .base_volume_renderer import VolumeRenderer
from ..deformers import SNARFDeformer, SMPLXDeformer
from ..renderers import GRenderer, get_covariance, batch_rodrigues
from ..superres import SuperresolutionHybrid2X, SuperresolutionHybrid4X
from lib.ops import SHEncoder, TruncExp, all_equal

# def get_embedder(multires):
#     embed_kwargs = {
#         'include_input': True,
#         'input_dims': 3,
#         'max_freq_log2': multires-1,
#         'num_freqs': multires,
#         'log_sampling': True,
#         'periodic_fns': [torch.sin, torch.cos],
#     }

#     embedder_obj = Embedder(**embed_kwargs)
#     def embed(x, eo=embedder_obj): return eo.embed(x)
#     return embed, embedder_obj.out_dim

# """ Positional encoding embedding. Code was taken from https://github.com/bmild/nerf. """
# class Embedder:
#     def __init__(self, **kwargs):
#         self.kwargs = kwargs
#         self.create_embedding_fn()

#     def create_embedding_fn(self):
#         embed_fns = []
#         d = self.kwargs['input_dims']
#         out_dim = 0
#         if self.kwargs['include_input']:
#             embed_fns.append(lambda x: x)
#             out_dim += d

#         max_freq = self.kwargs['max_freq_log2']
#         N_freqs = self.kwargs['num_freqs']

#         if self.kwargs['log_sampling']:
#             freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
#         else:
#             freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

#         for freq in freq_bands:
#             for p_fn in self.kwargs['periodic_fns']:
#                 embed_fns.append(lambda x, p_fn=p_fn,
#                                  freq=freq: p_fn(x * freq))
#                 out_dim += d

#         self.embed_fns = embed_fns
#         self.out_dim = out_dim

#     def embed(self, inputs):
#         return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

class PositionalEncoding():
    def __init__(self, input_dims=5, num_freqs=1, include_input=True):
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
class UVDecoder(VolumeRenderer):

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
                #  shape_layers=[3 * 32, 128],
                 density_layers=[128, 1],
                 color_layers=[128, 128, 3],
                 offset_layers=[128, 3],
                 scale_layers=[128, 3],
                 radius_layers=[128, 3],
                 use_dir_enc=True,
                 dir_layers=None,
                 scene_base_size=None,
                 scene_rand_dims=(0, 1),
                 activation='silu',
                #  sigma_activation='trunc_exp',
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
        # self.deformer = SNARFDeformer(gender)
        self.deformer = SMPLXDeformer(gender)
        # self.deformer = FLAMEDeformer()
        # self.renderer = GRenderer(image_size=128)
        self.renderer = GRenderer(image_size=image_size, bg_color=bg_color, f=5000)
        if superres:
            # self.superres = SuperresolutionHybrid2X(32, 32, 512)
            self.superres = None
            # self.superres = SuperresolutionHybrid2X(12, 32, 256)
            # self.superres = SuperresolutionHybrid4X(32, 32, 512)
        else:
            self.superres = None

        # self.embed_fn = None

        select_uv = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_uv_smplx_thu.npy'))
        self.register_buffer('select_coord', select_uv.unsqueeze(0)*2.-1.)

        init_pcd = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_smplx_thu.npy'))
        # init_pcd = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_flame.npy'))
        self.register_buffer('init_pcd', init_pcd.unsqueeze(0), persistent=False)
        self.num_init = self.init_pcd.shape[1]

        self.multires = multires
        if multires > 0:
            # init_uv = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_uv_new_smplx.npy'))
            # self.register_buffer('init_uv', init_uv.unsqueeze(0)*2.-1.)
            # self.pe = PositionalEncoding()
            init_pcd_nor = self.normalize(init_pcd.clone())
            input_coord = torch.cat([select_uv, init_pcd_nor], dim=-1)
            self.register_buffer('input_freq', input_coord.unsqueeze(0), persistent=False)
            # input_freq = self.pe.embed(input_coord)
            # self.register_buffer('input_freq', input_freq.unsqueeze(0), persistent=False)

            # base_layers[0] += self.pe.out_dim
            # color_layers[0] += self.pe.out_dim
            base_layers[0] += 5
            color_layers[0] += 5
        else:
            self.init_uv = None

        activation_layer = self.activation_dict[activation.lower()]

        # self.sig_density = nn.Sigmoid()
        # self.sig_color = nn.Sigmoid()

        base_net = [] # linear (in=18, out=64, bias=True)
        for i in range(len(base_layers) - 1):
            # base_net.append(nn.Linear(base_layers[i], base_layers[i + 1]))
            base_net.append(nn.Conv1d(base_layers[i], base_layers[i + 1], 1))
            if i != len(base_layers) - 2:
                base_net.append(nn.BatchNorm1d(base_layers[i+1]))
                base_net.append(activation_layer())
        self.base_net = nn.Sequential(*base_net)
        self.base_bn = nn.BatchNorm1d(base_layers[-1])
        self.base_activation = activation_layer()

        density_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(density_layers) - 1):
            # density_net.append(nn.Linear(density_layers[i], density_layers[i + 1]))
            density_net.append(nn.Conv1d(density_layers[i], density_layers[i + 1], 1))
            if i != len(density_layers) - 2:
                density_net.append(nn.BatchNorm1d(density_layers[i+1]))
                density_net.append(activation_layer())
        density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.density_net = nn.Sequential(*density_net)

        # scale_net = []
        # for i in range(len(scale_layers) - 1):
        #     scale_net.append(nn.Conv1d(scale_layers[i], scale_layers[i + 1], 1))
        #     if i != len(scale_layers) - 2:
        #         scale_net.append(nn.BatchNorm1d(scale_layers[i+1]))
        #         scale_net.append(activation_layer())
        # scale_net.append(nn.Sigmoid())
        # self.scale_net = nn.Sequential(*scale_net)

        # radius_net = []
        # for i in range(len(radius_layers) - 1):
        #     radius_net.append(nn.Conv1d(radius_layers[i], radius_layers[i + 1], 1))
        #     if i != len(radius_layers) - 2:
        #         radius_net.append(nn.BatchNorm1d(radius_layers[i+1]))
        #         radius_net.append(activation_layer())
        # radius_net.append(nn.Sigmoid())
        # self.radius_net = nn.Sequential(*radius_net)

        offset_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(offset_layers) - 1):
            # density_net.append(nn.Linear(density_layers[i], density_layers[i + 1]))
            offset_net.append(nn.Conv1d(offset_layers[i], offset_layers[i + 1], 1))
            if i != len(offset_layers) - 2:
                offset_net.append(nn.BatchNorm1d(offset_layers[i+1]))
                offset_net.append(activation_layer())
        # density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.offset_net = nn.Sequential(*offset_net)

        self.dir_net = None
        color_net = [] # linear(in=64, out=3, bias=True), sigmoid
        # if use_dir_enc:
        #     if dir_layers is not None:
        #         dir_net = []
        #         for i in range(len(dir_layers) - 1):
        #             dir_net.append(nn.Linear(dir_layers[i], dir_layers[i + 1]))
        #             if i != len(dir_layers) - 2:
        #                 dir_net.append(activation_layer())
        #         self.dir_net = nn.Sequential(*dir_net)
        #     else:
        #         color_layers[0] = color_layers[0] + 16  # sh_encoding
        for i in range(len(color_layers) - 1):
            # color_net.append(nn.Linear(color_layers[i], color_layers[i + 1]))
            color_net.append(nn.Conv1d(color_layers[i], color_layers[i + 1], kernel_size=1))
            if i != len(color_layers) - 2:
                color_net.append(nn.BatchNorm1d(color_layers[i+1]))
                color_net.append(activation_layer())
        color_net.append(nn.Sigmoid())
        # color_net.append(nn.Conv2d(color_layers[-2], color_layers[-1], kernel_size=1))
        self.color_net = nn.Sequential(*color_net)
        self.code_dropout = nn.Dropout2d(code_dropout) if code_dropout > 0 else None

        self.flip_z = flip_z
        self.extend_z = extend_z

        # dist2 = torch.clamp_min(distCUDA2(init_pcd.cuda()), 0.0000001)
        # scales = torch.sqrt(dist2)[...,None].repeat(1, 3)
        # self.register_buffer('scales', scales)

        # rots = torch.zeros((init_pcd.shape[0], 4))

        init_rot = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_rot_smplx_thu.npy'))
        # init_rot = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_rot_flame.npy'))
        self.register_buffer('init_rot', init_rot, persistent=False)

        face_mask = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/debug/face_mask_thu.npy'))
        self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

        hands_mask = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/debug/hands_mask_thu.npy'))
        self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

        # rots[:, 0] = 1
        # self.register_buffer('rots', rots)

        # opacity = torch.ones((init_pcd.shape[0], 3))
        # self.register_buffer('opacity', opacity)

        self.init_weights()

    def normalize(self, value):
        "normalize the value into [-1, 1]"
        print(value.shape)
        value_min, _ = value.min(0, keepdim=True)
        value_max, _ = value.max(0, keepdim=True)
        value_nor = ((value - value_min) / (value_max - value_min)) * 2 - 1
        return value_nor
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        if self.dir_net is not None:
            constant_init(self.dir_net[-1], 0)
        if self.offset_net is not None:
            self.offset_net[-1].weight.data.uniform_(-1e-5, 1e-5)
            self.offset_net[-1].bias.data.zero_()
        # nn.init.constant_(self.color_net[-2].weight, 0)
        # nn.init.constant_(self.color_net[-2].bias, 0)
        # nn.init.constant_(self.scale_net[0].weight, 0)
        # nn.init.constant_(self.scale_net[0].bias, 0)
        # nn.init.constant_(self.radius_net[0].weight, 0)
        # nn.init.constant_(self.radius_net[0].bias, 0)
        # if self.density_net is not None:
            # nn.init.constant_(self.density_net[0].bias, 2.1972) # 0.9
            # nn.init.constant_(self.density_net[0].bias, -1.3863) # 0.2
            # nn.init.constant_(self.density_net[0].bias, 0)

    def xyz_transform(self, xyz, smpl_params=None, num_scenes=1):
        if xyz.dim() != 3:
            print(xyz.shape)
            assert False
        self.deformer.prepare_deformer(smpl_params, num_scenes, device=xyz.device)
        xyz, tfs = self.deformer(xyz, cano=False)
        return xyz, tfs

    def extract_pcd(self, code, smpl_params, init=False):
        num_scenes, n_channels, h, w = code.size()
        init_pcd = self.init_pcd.repeat(num_scenes, 1, 1)
        
        if self.code_dropout is not None:
            code = self.code_dropout(
                code.reshape(num_scenes, n_channels, h, w)
            ).reshape(num_scenes, n_channels, h, w)
            assert False
        if self.scene_base is not None:
            code = code + self.scene_base
            assert False
        
        sigmas, rgbs, radius, rot, offset = self._decode(code, init=init)
        canon_pcd = init_pcd + offset
        # canon_pcd = init_pcd
        defm_pcd, tfs = self.xyz_transform(canon_pcd, smpl_params, num_scenes)
        
        return defm_pcd, sigmas, rgbs, offset, radius, tfs, rot
    
    def _decode(self, point_code, init=False):
        if point_code.dim() == 4:
            num_scenes, n_channels, h, w = point_code.shape
            select_coord = self.select_coord.unsqueeze(1).repeat(num_scenes, 1, 1, 1)
            # init_coord = self.init_pcd.repeat(num_scenes, 1, 1).permute(0, 2, 1)
            if self.multires != 0:
                input_freq = self.input_freq.repeat(num_scenes, 1, 1).permute(0, 2, 1)
            # uv_coord = self.init_uv if self.init_uv is None else self.init_uv.repeat(num_scenes, 1, 1)
        else:
            assert False
        
        # geo_code, tex_code = point_code.split(16, dim=1)
        # coord_in = geo_code if self.init_uv is None else torch.cat([geo_code, coord, uv_coord], dim=-1)
        code_in = F.grid_sample(point_code, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, n_channels, -1)
        # select_coord = select_coord.permute(0, 1, 3, 2)
        # geo_code, tex_code = point_code.reshape(num_scenes, n_channels, -1).split(16, dim=1)
        geo_code, tex_code = code_in.reshape(num_scenes, n_channels, -1).split(16, dim=1)
        # tex_code = torch.flip(tex_code, [0])
        # base_in = geo_code if self.multires == 0 else torch.cat([geo_code, init_coord, select_coord[:, 0]], dim=1)
        base_in = geo_code if self.multires == 0 else torch.cat([geo_code, input_freq], dim=1)
        base_x = self.base_net(base_in)
        base_x_act = self.base_activation(self.base_bn(base_x))
        sigma = self.density_net(base_x_act).permute(0, 2, 1)
        offset = self.offset_net(base_x_act).permute(0, 2, 1)

        # color_in = torch.cat([base_x, tex_code], dim=1) if self.multires == 0 else torch.cat([base_x, tex_code, init_coord, select_coord[:, 0]], dim=1)
        # color_in = tex_code if self.multires == 0 else torch.cat([tex_code, init_coord, select_coord[:, 0]], dim=1)
        color_in = tex_code if self.multires == 0 else torch.cat([tex_code, input_freq], dim=1)
        # radius = self.scale_net(color_in).permute(0, 2, 1)
        # rot = self.radius_net(color_in).permute(0, 2, 1)
        rgbs_radius_rot = self.color_net(color_in).permute(0, 2, 1)
        rgbs, radius, rot = rgbs_radius_rot.split([3, 3, 3], dim=2)
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
        # rgbs = rgbs * 0 + 0.5
        # radius = radius.clip(-1, 1)
        radius = (radius - 0.5) * 2
        rot = (rot - 0.5) * np.pi
        # rot = rot.clip(-np.pi/2, np.pi/2)

        return sigma, rgbs, radius, rot, offset

    def gaussian_render(self, pcd, sigmas, rgbs, normals, cov3D, num_scenes, num_imgs, cameras, use_scale=False, radius=None, return_norm=False):
    # def gaussian_render(self, pcd, sigmas, rgbs, num_scenes, num_imgs, num_points, cameras):
        #TODO: add mask or visible points to images or select ind to images
        # mask = sigmas > 0.5
        assert num_scenes == 1
        
        pcd = pcd.reshape(-1, 3)
        if use_scale:
            # dist2 = torch.clamp_min(distCUDA2(self.init_pcd[0]), 0.0000001)
            dist2 = torch.clamp_min(distCUDA2(pcd), 0.0000001)
            # scales = (0.6 * torch.sqrt(dist2)[...,None].repeat(1, 3)).detach()
            scales = torch.sqrt(dist2)[...,None].repeat(1, 3).detach()
            scale = (radius+1)*scales
            cov3D = get_covariance(scale, cov3D).reshape(-1, 6)
        # feat = torch.cat([rgbs, normals], dim=-1)
        images_all = []
        norm_all = [] if return_norm else None

        # R_delta = batch_rodrigues(rot)
        # R = torch.bmm(self.init_rot[masks], R_delta)
        # R_def = torch.bmm(tfs[:,:3,:3], R)
        # scale = (radius+1)*self.scales[masks]
        # cov3D = get_covariance(scale, R_def)

        if return_norm:
            for i in range(num_imgs):
                self.renderer.prepare(cameras[i])
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
                image = self.renderer.render_gaussian(means3D=pcd, colors_precomp=rgbs, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                images_all.append(image)
           
        images_all = torch.stack(images_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        
        return images_all, norm_all

    # def point_density_decode(self, xyzs, code, smpl_params=None, **kwargs):
    #     sigmas, _, offset, radius, num_points = self.point_decode(
    #         xyzs, None, code, smpl_params=smpl_params, density_only=True, **kwargs)
    #     return sigmas, offset, radius, num_points

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        num_scenes, num_chn, h, w = code.size()
        code_viz = code.reshape(num_scenes, 4, 8, h, w).cpu().numpy()
        if not self.flip_z:
            code_viz = code_viz[..., ::-1, :]
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 4 * h, 8 * w)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])

    def forward(self, code, grid_size, smpl_params, cameras, num_imgs,
                dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False, return_norm=False, init=False):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        num_scenes = len(code)
        assert num_scenes > 0

        if self.training:
            image = []
            norm = [] if return_norm else None
            # offset = []
            # scales = []
            # part_mask = []
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=init)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            # R = self.init_rot.repeat(num_scenes, 1, 1)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            # scale = (radius.reshape(-1, 3)+1)*self.scales.repeat(num_scenes, 1)
            # cov3D = get_covariance(scale, R_def).reshape(num_scenes, -1, 6)
            normals = (R_def[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            # for camera_single, cov3D_single, pcd_single, rgbs_single, sigmas_single, normal_single in zip(cameras, cov3D, xyzs, rgbs, sigmas, normals):
                # image_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, cov3D_single, 1, num_imgs, camera_single)
            if return_norm:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, norm_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, return_norm=True)
                    image.append(image_single)
                    norm.append(norm_single)
                norm = torch.cat(norm, dim=0)
            else:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single)
                    image.append(image_single)
            #     offset.append(offset_single)
            #     part_mask.append(self.part_mask[mask_single])
            #     scales.append(scale_single)
            
            image = torch.cat(image, dim=0)
            # offset = torch.cat(offset, dim=0)
            # scales = torch.cat(scales, dim=0)
            # part_mask = torch.cat(part_mask, dim=0)
            offset_dist = offsets ** 2
            weighted_offset = torch.mean(offset_dist) + torch.mean(offset_dist[self.hands_mask.repeat(num_scenes, 1)]) + torch.mean(offset_dist[self.face_mask.repeat(num_scenes, 1)])

            results = dict(
                # part_mask=part_mask,
                # scales=scale.reshape(num_scenes, -1, 3),
                norm=norm,
                image=image,
                offset=weighted_offset)
        else:
            device = code.device
            dtype = torch.float32

            image = []
            offsets = None
            scale = None
            # part_mask = None

            xyzs, sigmas, rgbs, offsets, radius, tfs, rot = self.extract_pcd(code, smpl_params, init=False)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            # print(tfs.flatten(0, 1)[12, :3, :3])
            # print(tfs.flatten(0, 1).inverse().transpose(-2,-1)[12, :3, :3])
            # assert False
            # scale = (radius.reshape(-1, 3)+1)*self.scales.repeat(num_scenes, 1)
            # cov3D = get_covariance(scale, R_def).reshape(num_scenes, -1, 6)
            normals = (R_def[:, :, -1] * 0.5 + 0.5).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            # for camera_single, cov3D_single, pcd_single, rgbs_single, sigmas_single, normal_single in zip(cameras, cov3D, xyzs, rgbs, sigmas, normals):
                # image_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, cov3D_single, 1, num_imgs, camera_single)
            for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                image_single, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single)
                image.append(image_single)

            image = torch.cat(image, dim=0)

            results = dict(
                # part_mask=part_mask,
                # scales=scale.reshape(num_scenes, -1, 3),
                image=image,
                offset=offsets)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results
