import torch
import torch.nn as nn
import numpy as np

from mmgen.models.builder import build_module

from lib.ops import (
    batch_near_far_from_aabb,
    march_rays_train, batch_composite_rays_train, march_rays, composite_rays)


class VolumeRenderer(nn.Module):
    def __init__(self,
                 bound=1,
                 min_near=0.2,
                 bg_radius=-1,
                 max_steps=256,
                 decoder_reg_loss=None,
                 ):
        super().__init__()

        self.bound = bound
        self.min_near = min_near
        self.bg_radius = bg_radius  # radius of the background sphere.
        self.max_steps = max_steps
        self.decoder_reg_loss = build_module(decoder_reg_loss) if decoder_reg_loss is not None else None

        # prepare aabb with a 6D tensor (xmin, ymin, zmin, xmax, ymax, zmax)
        aabb = torch.FloatTensor([-bound, -bound, -bound, bound, bound, bound])
        self.register_buffer('aabb', aabb)

    def extract_pcd(self, num_scenes, code):
        raise NotImplementedError
        
    def point_decode(self, xyzs, dirs, code):
        raise NotImplementedError

    def point_density_decode(self, xyzs, code):
        raise NotImplementedError

    def loss(self):
        assert self.decoder_reg_loss is None
        return None
    
    def point_render(self, pcd, mask, rgbs, num_scenes, num_points, cameras):
        raise NotImplementedError
    
    def gaussian_render(self, pcd, mask, rgbs, num_scenes, num_points, cameras):
        raise NotImplementedError

    # def forward(self, code, density_bitfield, grid_size, smpl_params, cameras, num_imgs,
    #             dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False):
    def forward(self, code, grid_size, smpl_params, cameras, num_imgs,
                points=None, masks=None, dt_gamma=0, perturb=False, T_thresh=1e-4, return_loss=False, stage2=True, init=False):
        """
        Args:
            rays_o: Shape (num_scenes, num_rays_per_scene, 3)
            rays_d: Shape (num_scenes, num_rays_per_scene, 3)
            code: Shape (num_scenes, *code_size)
            density_bitfield: Shape (num_scenes, griz_size**3 // 8)
        """
        # num_scenes = len(rays_o)
        num_scenes = len(code)
        assert num_scenes > 0
        if isinstance(grid_size, int):
            grid_size = [grid_size] * num_scenes
        if isinstance(dt_gamma, float):
            dt_gamma = [dt_gamma] * num_scenes

        # nears, fars = batch_near_far_from_aabb(rays_o, rays_d, self.aabb, self.min_near)

        if self.training:
            # weights_sum = []
            image = []
            offset = []
            scales = []
            # front_masks = []
            # xyzs = []
            # dirs = []
            # deltas = []
            # rays = []
            
            # for (rays_o_single, rays_d_single, density_bitfield_single,
            #      nears_single, fars_single, grid_size_single, dt_gamma_single) in zip(
            #         rays_o, rays_d, density_bitfield, nears, fars, grid_size, dt_gamma):
            # for (density_bitfield_single, grid_size_single, dt_gamma_single) in zip(
            #         density_bitfield, grid_size, dt_gamma):
            # for smpl_param_single in smpl_params:
            # for i in range(num_scenes):
                # xyzs_single, dirs_single, deltas_single, rays_single = march_rays_train(
                #     self.bound, density_bitfield_single,
                #     1, grid_size_single,
                #     perturb=perturb, align=128, force_all_rays=True,
                #     dt_gamma=dt_gamma_single.item(), max_steps=self.max_steps)
                # xyzs_single = extract_pcd(smpl_param_single)
                # xyzs.append(xyzs_single)
                # dirs.append(dirs_single)
                # deltas.append(deltas_single)
                # rays.append(rays_single)
            # sigmas, rgbs, num_points = self.point_decode(xyzs, dirs, code, smpl_params)
            # for i in range(num_scenes):
            for smpl_param_single, camera_single, code_single, point_single, mask_single in zip(smpl_params, cameras, code, points, masks):
                xyzs, sigmas, rgbs, num_points, offset_single, radius, tfs, rot = self.extract_pcd(1, code_single[None], smpl_param_single[None], point_single[None], mask_single[None], stage2=stage2, init=init)
                # xyzs, sigmas, rgbs, num_points, offset_single = self.extract_pcd(1, code_single[None], smpl_param_single[None])
                # image_single, weights_sum_single = self.point_render(xyzs, sigmas, rgbs, 1, num_imgs, num_points, [camera_single])
                # front_mask = torch.ones_like(sigmas).unsqueeze(-1)
                # rgb_feat = torch.cat([rgbs, front_mask], dim=-1)
                image_single, scale_single = self.gaussian_render(xyzs, sigmas, rgbs, radius, tfs, rot, 1, num_imgs, num_points, camera_single, mask_single, stage2=stage2)
                # image_single, mask_single = image_mask.split([3, 1], dim=-1)
                # image_single = self.gaussian_render(xyzs, sigmas, rgbs, 1, num_imgs, num_points, [camera_single])
                # weights_sum.append(weights_sum_single)
                # front_masks.append(mask_single)
                image.append(image_single)
                # offset.append(offset_single.unsqueeze(0))
                offset.append(offset_single)
                part_mask.append(self.part_mask[mask_single])
                scales.append(scale_single)
            # weights_sum = torch.cat(weights_sum, dim=0)
            image = torch.cat(image, dim=0)
            # front_masks = torch.cat(front_masks, dim=0)
            offset = torch.cat(offset, dim=0)
            scales = torch.cat(scales, dim=0)
            part_mask = torch.cat(part_mask, dim=0)
            # sigmas, rgbs, num_points = self.point_decode(xyzs, code, smpl_params)
            # weights_sum, depth, image = batch_composite_rays_train(sigmas, rgbs, deltas, rays, num_points, T_thresh)

        else:
            # device = rays_o.device
            device = code.device
            dtype = torch.float32

            # weights_sum = []
            # depth = []
            image = []
            offset = None
            scales = None
            part_mask = None
            # front_masks = None

            # for (rays_o_single, rays_d_single,
            #      code_single, density_bitfield_single,
            #      nears_single, fars_single,
            #      grid_size_single, dt_gamma_single) in zip(
            #         rays_o, rays_d, code, density_bitfield, nears, fars, grid_size, dt_gamma):
            # for smpl_param_single in smpl_params:
            for smpl_param_single, camera_single, code_single, point_single, mask_single in zip(smpl_params, cameras, code, points, masks):
                xyzs, sigmas, rgbs, num_points, _, radius, tfs, rot = self.extract_pcd(1, code_single[None], smpl_param_single[None], point_single[None], mask_single[None], stage2=stage2)
                # print((sigmas>0.2).sum())
                # print((sigmas>0.1).sum())
                # print((sigmas>0.05).sum())
                # assert False
                # xyzs, sigmas, rgbs, num_points, _ = self.extract_pcd(1, code_single[None], smpl_param_single[None])
                # image_single, weights_sum_single = self.point_render(xyzs, sigmas, rgbs, 1, num_imgs, num_points, [camera_single])
                # print(len(camera_single))
                # print(camera_single[0].shape)
                # print(camera_single[1].shape)
                # assert False
                image_single, _ = self.gaussian_render(xyzs, sigmas, rgbs, radius, tfs, rot, 1, num_imgs, num_points, camera_single, mask_single)
                # image_single = self.gaussian_render(xyzs, sigmas, rgbs, 1, num_imgs, num_points, [camera_single])
                # num_rays_per_scene = rays_o_single.size(0)
                # num_rays_per_scene = num_imgs * 128 * 128

                # weights_sum_single = torch.zeros(num_rays_per_scene, dtype=dtype, device=device)
                # depth_single = torch.zeros(num_rays_per_scene, dtype=dtype, device=device)
                # image_single = torch.zeros(num_rays_per_scene, 3, dtype=dtype, device=device)

                # num_rays_alive = num_rays_per_scene
                # rays_alive = torch.arange(num_rays_alive, dtype=torch.int32, device=device)  # (num_rays_alive,)
                # rays_t = nears_single.clone()  # (num_rays_alive,)

                # step = 0
                # while step < self.max_steps:
                    # count alive rays
                    # num_rays_alive = rays_alive.size(0)
                    # exit loop
                    # if num_rays_alive == 0:
                    #     break
                    # decide compact_steps
                    # n_step = min(max(num_rays_per_scene // num_rays_alive, 1), 8)
                    # xyzs, dirs, deltas = march_rays(
                    #     num_rays_alive, n_step, rays_alive, rays_t, rays_o_single, rays_d_single,
                    #     self.bound, density_bitfield_single, 1, grid_size_single, nears_single, fars_single,
                    #     align=128, perturb=perturb, dt_gamma=dt_gamma_single.item(), max_steps=self.max_steps)
                    # sigmas, rgbs, _ = self.point_decode([xyzs], [dirs], code_single[None], smpl_params)
                    # composite_rays(num_rays_alive, n_step, rays_alive, rays_t, sigmas, rgbs, deltas,
                    #                weights_sum_single, depth_single, image_single, T_thresh)
                    # rays_alive = rays_alive[rays_alive >= 0]
                    # step += n_step

                # weights_sum.append(weights_sum_single)
                # depth.append(depth_single)
                image.append(image_single)

        # results = dict(
        #     weights_sum=weights_sum,
        #     depth=depth,
        #     image=image)
        results = dict(
            # weights_sum=weights_sum,
            # front_masks=front_masks,
            part_mask=part_mask,
            scales=scales,
            image=image,
            offset=offset)

        if return_loss:
            results.update(decoder_reg_loss=self.loss())

        return results
