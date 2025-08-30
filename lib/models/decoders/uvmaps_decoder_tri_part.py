import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from pytorch3d import ops

from mmcv.cnn import xavier_init, constant_init
from mmgen.models.builder import MODULES
import numpy as np
import time
import cv2
import math
from simple_knn._C import distCUDA2
from pytorch3d.transforms import quaternion_to_matrix

from .base_volume_renderer import VolumeRenderer
from ..deformers import SMPLXDeformer
from ..renderers import GRenderer, GFRenderer, get_covariance, batch_rodrigues
# from ..renderers import GRenderer, get_covariance, batch_rodrigues
from ..superres import SuperresolutionHybrid2X, SuperresolutionHybrid4X
from lib.ops import TruncExp


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
class UVTPDecoder(VolumeRenderer):

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
        torch.autograd.set_detect_anomaly(True)
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
        self.deformer = SMPLXDeformer(gender)
        pcd_label = torch.as_tensor(np.load('work_dirs/cache/part_label_full.npy'))
        self.register_buffer('pcd_label', pcd_label, persistent=False)
        if image_size == 1024:
            self.renderer = GFRenderer(image_size=image_size, bg_color=bg_color, f=5000, label=self.pcd_label)
        elif image_size == 512:
            self.renderer = GFRenderer(image_size=image_size, bg_color=bg_color, f=2500, label=self.pcd_label)
        else:
            assert False
        if superres:
            self.superres = None
        else:
            self.superres = None

        select_uv = torch.as_tensor(np.load('work_dirs/cache/init_uv_smplx_full.npy'))
        self.register_buffer('select_coord', select_uv.unsqueeze(0)*2.-1., persistent=False)

        init_pcd = torch.as_tensor(np.load('work_dirs/cache/init_pcd_smplx_full.npy'))
        self.num_init = init_pcd.shape[1]

        self.init_uv = None

        activation_layer = self.activation_dict[activation.lower()]

        base_net = [] # linear (in=18, out=64, bias=True)
        for i in range(len(base_layers) - 1):
            base_net.append(nn.Conv2d(base_layers[i], base_layers[i + 1], 7, padding=3))
            if i != len(base_layers) - 2:
                base_net.append(nn.BatchNorm2d(base_layers[i+1]))
                base_net.append(activation_layer())
        self.base_net = nn.Sequential(*base_net)
        self.base_bn = nn.BatchNorm2d(base_layers[-1])
        self.base_activation = activation_layer()

        density_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(density_layers) - 1):
            density_net.append(nn.Conv2d(density_layers[i], density_layers[i + 1], 1))
            if i != len(density_layers) - 2:
                density_net.append(nn.BatchNorm2d(density_layers[i+1]))
                density_net.append(activation_layer())
        density_net.append(self.activation_dict[sigma_activation.lower()]())
        self.density_net = nn.Sequential(*density_net)

        offset_net = [] # linear(in=64, out=1, bias=True), sigmoid
        for i in range(len(offset_layers) - 1):
            offset_net.append(nn.Conv2d(offset_layers[i], offset_layers[i + 1], 1))
            if i != len(offset_layers) - 2:
                offset_net.append(nn.BatchNorm2d(offset_layers[i+1]))
                offset_net.append(activation_layer())
        self.offset_net = nn.Sequential(*offset_net)

        self.dir_net = None
        tex_net = [] # linear(in=64, out=3, bias=True), sigmoid
        color_net = []
        cov_net = []
        for i in range(len(color_layers) - 2):
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

        init_rot = torch.as_tensor(np.load('work_dirs/cache/init_rot_smplx_full.npy'))
        self.register_buffer('init_rot', init_rot, persistent=False)

        face_mask = torch.as_tensor(np.load('work_dirs/cache/face_mask_body.npy'))
        self.register_buffer('face_mask', face_mask.unsqueeze(0), persistent=False)

        hands_mask = torch.as_tensor(np.load('work_dirs/cache/hands_mask_body.npy'))
        self.register_buffer('hands_mask', hands_mask.unsqueeze(0), persistent=False)

        outside_mask = torch.as_tensor(np.load('work_dirs/cache/outside_mask_body.npy'))
        self.register_buffer('outside_mask', outside_mask.unsqueeze(0), persistent=False)

        nose_mask = torch.as_tensor(np.load('work_dirs/cache/nose_mask_body.npy'))
        self.register_buffer('nose_mask', nose_mask.unsqueeze(0), persistent=False)

        offset_mask = (self.face_mask + self.outside_mask) * (~self.nose_mask)
        self.register_buffer('offset_mask', offset_mask, persistent=False)

        mask_other = self.pcd_label != 1
        normals = self.init_rot[:, :, -1]
        init_pcd[mask_other] += 1e-2 * normals[mask_other]
        self.register_buffer('init_pcd', init_pcd.unsqueeze(0), persistent=False)
        self.register_buffer('mask_other', mask_other, persistent=False)

        attr_weight = torch.tensor([1, 2, 2, 2, 0, 0, 0, 0, 0, 0, 2, 2, 2])
        self.register_buffer('attr_weight', attr_weight.reshape(1, -1, 1, 1), persistent=False)

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

    def xyz_transform(self, xyz, smpl_params=None, num_scenes=1):
        if xyz.dim() != 3:
            print(xyz.shape)
            assert False
        self.deformer.prepare_deformer(smpl_params, num_scenes, device=xyz.device)
        xyz, tfs = self.deformer(xyz, mask=(self.face_mask+self.hands_mask+self.outside_mask), cano=False)
        return xyz, tfs

    def extract_pcd(self, code, smpl_params, init=False, composite=False):
        if isinstance(code, list):
            num_scenes, _, h, w = code[0].size()
        elif code.dim() == 5:
            num_scenes, n_layer, n_channels, h, w = code.size()
        else:
            num_scenes, n_channels, h, w = code.size()
        if composite:
            num_scenes = 1
        
        init_pcd = self.init_pcd.expand(num_scenes, -1, -1)
        sigmas, rgbs, radius, rot, offset, attrmap = self._decode(code, init=init, composite=composite)

        hands_color = rgbs[:, :self.hands_mask.shape[1]][:, self.face_mask[0]]
        mean_body_color = hands_color.mean(1)
        canon_pcd = init_pcd + offset
        defm_pcd, tfs = self.xyz_transform(canon_pcd, smpl_params, num_scenes)
        
        return defm_pcd, sigmas, rgbs, offset, radius, tfs, rot, mean_body_color, attrmap
    
    def _decode(self, point_code, init=False, composite=False):
        assert point_code.dim() == 4
        num_scenes, n_channels, h, w = point_code.shape
        select_coord = self.select_coord.unsqueeze(1).expand(num_scenes, -1, -1, -1)

        geo_code, tex_code = point_code.split(6, dim=1)
        base_x = self.base_net(geo_code)
        base_x_act = self.base_activation(self.base_bn(base_x))
        sigma = self.density_net(base_x_act)
        offset = self.offset_net(base_x_act)
        base_tex = self.tex_net(tex_code)
        rgbs = self.color_net(base_tex)
        radius_rot = self.cov_net(base_tex)
        outputs = torch.cat([sigma, offset, rgbs, radius_rot], dim=1) # [1, 13, 256, 256*3]
        
        # select_coord [1, 1, 89490, 2]
        # output_attr [1, 13, 1, 89490] -> [1, 13, 89490] -> [1, 89490, 13]
        output_attr = F.grid_sample(outputs, select_coord, mode=self.interp_mode, padding_mode='border', align_corners=False).reshape(num_scenes, 13, -1).permute(0, 2, 1)

        if composite:
            body_mask = self.pcd_label == 1
            cloth_mask = self.pcd_label == 2
            hair_mask = self.pcd_label == 3
            shoes_mask = self.pcd_label == 4
            bottom_mask = self.pcd_label == 5
        
            _, idx_bottom, _ = ops.knn_points(self.init_pcd[:, bottom_mask], self.init_pcd[:, body_mask])
            _, idx_top, _ = ops.knn_points(self.init_pcd[:, cloth_mask], self.init_pcd[:, body_mask])
            _, idx_shoes, _ = ops.knn_points(self.init_pcd[:, shoes_mask], self.init_pcd[:, body_mask])

            wo_mask = torch.logical_or(self.hands_mask, self.offset_mask)[0]
            body_mask *= torch.cat([~wo_mask, torch.zeros(89490-34858).to(body_mask.device).bool()])
            output_attr[0, body_mask, 1:4] = output_attr[3, body_mask, 1:4]
            output_attr[0, idx_shoes[0], 1:4] = output_attr[4, idx_shoes[0], 1:4]
            output_attr[0, idx_top[0], 1:4] = output_attr[1, idx_top[0], 1:4]
            output_attr[0, idx_bottom[0], 1:4] = output_attr[2, idx_bottom[0], 1:4]

            output_attr[0, cloth_mask] = output_attr[1, cloth_mask]
            output_attr[0, bottom_mask] = output_attr[2, bottom_mask]
            output_attr[0, hair_mask] = output_attr[3, hair_mask]
            output_attr[0, shoes_mask] = output_attr[4, shoes_mask]
            output_attr = output_attr[:1]

        sigma, offset, rgbs, radius, rot = output_attr.split([1, 3, 3, 3, 3], dim=2)

        if self.sigmoid_saturation > 0:
            rgbs = rgbs * (1 + self.sigmoid_saturation * 2) - self.sigmoid_saturation

        radius = (radius - 0.5) * 2 ## [radius [-1, 1]]
        rot = (rot - 0.5) * np.pi

        return sigma, rgbs, radius, rot, offset, outputs

    def gaussian_render(self, pcd, sigmas, rgbs, normals, cov3D, num_scenes, num_imgs, cameras, use_scale=False, radius=None, return_norm=False, return_viz=False, mask=None):
        assert num_scenes == 1
        
        pcd = pcd.reshape(-1, 3)
 
        if use_scale:
            part_pcd = pcd.split([34858, 21424, 11056, 22152], dim=0)
            dist2 = torch.cat([torch.clamp_min(distCUDA2(pt), 0.0000001) for pt in part_pcd], dim=0)
            scales = torch.sqrt(dist2)[...,None].repeat(1, 3).detach()
            scales[:, -1] *= 0.01
            scale = (radius+1)*scales
            cov3D = get_covariance(scale, cov3D).reshape(-1, 6)

        images_all = []
        segs_all = []
        viz_masks = [] if return_viz else None
        norm_all = [] if return_norm else None

        if mask != None:
            pcd = pcd[mask]
            rgbs = rgbs[mask]
            sigmas = sigmas[mask]
            cov3D = cov3D[mask]
            normals = normals[mask]
        if return_norm:
            assert False
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
                image, seg = self.renderer.render_gaussian(means3D=pcd, colors_precomp=rgbs, 
                    rotations=None, opacities=sigmas, scales=None, cov3D_precomp=cov3D)
                images_all.append(image)
                segs_all.append(seg)
                if return_viz:
                    viz_mask = self.renderer.render_gaussian(means3D=pcd, colors_precomp=pcd.clone(), 
                        rotations=None, opacities=sigmas*0+1, scales=None, cov3D_precomp=cov3D)
                    viz_masks.append(viz_mask)
        images_all = torch.stack(images_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        segs_all = torch.stack(segs_all, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2)
        # if return_viz:
        #     viz_masks = torch.stack(viz_masks, dim=0).unsqueeze(0).permute(0, 1, 3, 4, 2).reshape(1, -1, 3)
        #     dist_sq, idx, neighbors = ops.knn_points(pcd.unsqueeze(0), viz_masks[:, ::10], K=1, return_nn=True)
        #     viz_masks = (dist_sq < 0.0005)[0]

        if use_scale:
            return images_all, norm_all, viz_masks, scale, segs_all
        else:
            return images_all, norm_all, viz_masks, segs_all

    def visualize(self, code, scene_name, viz_dir, code_range=[-1, 1]):
        # num_scenes, n_layer, num_chn, h, w = code.size()
        if code.dim() == 4:
            num_scenes, num_chn, h, w = code.size()
            code_ = code.reshape(num_scenes, num_chn, h, 3, w//3).permute(0, 3, 1, 2, 4)
            code_ = code_[:, :, ::2]
        elif code.dim() == 5:
            assert False
            num_scenes, n_layer, num_chn, h, w = code.size()
            code_ = code.flatten(1, 2)[:, ::2]

        code_viz = code_.cpu().numpy()
        if not self.flip_z:
            code_viz = code_viz[..., ::-1, :]
        code_viz = code_viz.transpose(0, 1, 3, 2, 4).reshape(num_scenes, 3 * h, 2 * w)
        for code_single, code_viz_single, scene_name_single in zip(code, code_viz, scene_name):
            torch.save(code_single, os.path.join(viz_dir, 'scene_' + scene_name_single + '.pth'))
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '.png'), code_viz_single,
                       vmin=code_range[0], vmax=code_range[1])

    def forward(self, code, grid_size, smpl_params, cameras, num_imgs,
                dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False, return_norm=False, init=False, mask=None, composite=False):
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
        if composite:
            num_scenes = 1
        assert num_scenes > 0

        code_h, code_w = code.shape[-2:]

        if self.training:
            image = []
            segs = []
            norm = [] if return_norm else None
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot, reg_color, attrmap = self.extract_pcd(code, smpl_params, init=init)
            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
            if return_norm:
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, norm_single, _, _, _ = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single, return_norm=True)
                    image.append(image_single)
                    norm.append(norm_single)
                norm = torch.cat(norm, dim=0)
            else:
                # rgbs, sigmas, radius
                for camera_single, R_def_single, pcd_single, rgbs_single, sigmas_single, normal_single, radius_single in zip(cameras, R_def_batch, xyzs, rgbs, sigmas, normals, radius):
                    image_single, _, _, _, seg_single = self.gaussian_render(pcd_single, sigmas_single, rgbs_single, normal_single, R_def_single, 1, num_imgs, camera_single, use_scale=True, radius=radius_single)
                    image.append(image_single)
                    segs.append(seg_single)
            
            image = torch.cat(image, dim=0)
            segs = torch.cat(segs, dim=0)
            offset_dist = (offsets ** 2).sum(-1)
            weighted_offset = torch.mean(offset_dist[:, ~self.mask_other]) + torch.mean(offset_dist[:, self.mask_other]) * 0.1 + torch.mean(offset_dist[:, :self.hands_mask.shape[1]][self.hands_mask.expand(num_scenes, -1)]) + torch.mean(offset_dist[:, :self.offset_mask.shape[1]][self.offset_mask.expand(num_scenes, -1)])

            attrmap = attrmap.reshape(num_scenes, -1, code_h, 3, code_w//3).permute(0, 3, 1, 2, 4).flatten(0, 1)
            attrmap = attrmap * self.attr_weight.expand(num_scenes*3, -1, -1, -1)
            results = dict(
                reg_color=reg_color,
                sigmas=sigmas,
                segs=segs,
                attrmap=attrmap.reshape(num_scenes, 39, code_h, code_w//3),
                norm=norm,
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
            xyzs, sigmas, rgbs, offsets, radius, tfs, rot, _, attrmap = self.extract_pcd(code, smpl_params, init=False, composite=composite)

            R_delta = batch_rodrigues(rot.reshape(-1, 3))
            R = torch.bmm(self.init_rot.repeat(num_scenes, 1, 1), R_delta)
            R_def = torch.bmm(tfs.flatten(0, 1)[:,:3,:3], R)
            normals = (R_def[:, :, -1]).reshape(num_scenes, -1, 3)
            R_def_batch = R_def.reshape(num_scenes, -1, 3, 3)
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
                viz_masks = None
            image = torch.cat(image, dim=0)
            segs = torch.cat(segs, dim=0)

            results = dict(
                segs=segs,
                viz_masks=viz_masks,
                scales=None,
                image=image,
                offset=offsets)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results