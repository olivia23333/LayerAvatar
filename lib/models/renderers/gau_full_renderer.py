from diff_Gauss import GaussianRasterizationSettings, GaussianRasterizer
import torch
import torch.nn as nn
import math
import cv2
import numpy as np
# from roma import rotmat_to_unitquat, quat_xyzw_to_wxyz
# from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

def batch_rodrigues(rot_vecs, epsilon = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat

def build_scaling_rotation(s, r, tfs):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=s.device)
    R = build_rotation(r)
    # tfs_rot = tfs[0,:,:3,:3]
    # tfs_rot = torch.eye(3, device=s.device).unsqueeze(0).repeat(s.shape[0], 1, 1)
    # R_ = tfs_rot @ R
    R_ = R

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R_ @ L
    return L

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation, tfs):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation, tfs)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=q.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def get_covariance(scaling, rotation, scaling_modifier = 1):
    L = torch.zeros_like(rotation)
    L[:, 0, 0] = scaling[:, 0]
    L[:, 1, 1] = scaling[:, 1]
    L[:, 2, 2] = scaling[:, 2]
    actual_covariance = rotation @ (L**2) @ rotation.permute(0, 2, 1)
    return strip_symmetric(actual_covariance)

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class GFRenderer(nn.Module):
    def __init__(self, image_size=256, anti_alias=False, f=5000, label=None, near=0.01, far=40, bg_color=0):
        super().__init__()

        self.anti_alias = anti_alias
        self.image_size = image_size
        self.tanfov = self.image_size / (2 * f)

        if bg_color == 0:
            bg = torch.tensor([0, 0, 0], dtype=torch.float32)
        else:
            bg = torch.tensor([1, 1, 1], dtype=torch.float32)

        self.register_buffer('bg', bg)
        
        opengl_proj = torch.tensor([[2 * f / self.image_size, 0.0, 0.0, 0.0],
                                    [0.0, 2 * f / self.image_size, 0.0, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).float().unsqueeze(0).transpose(1, 2)
        self.register_buffer('opengl_proj', opengl_proj, persistent=False)

        part_mask = torch.cat([label == (i+1) for i in range(5)], dim=0)
        self.register_buffer('part_label', label, persistent=False)
        self.register_buffer('part_mask', part_mask, persistent=False)

        if anti_alias: image_size = image_size*2
        
    def prepare(self, cameras):
        cam_center = cameras[:3]
        w2c = cameras[3:].reshape(4, 4)
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(self.opengl_proj)
        self.full_proj = full_proj
        self.w2c = w2c

        self.raster_settings = GaussianRasterizationSettings(
            image_height=self.image_size,
            image_width=self.image_size,
            tanfovx=self.tanfov,
            tanfovy=self.tanfov,
            bg=self.bg,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False,
            debug=False
        )
        raster_settings_bg = GaussianRasterizationSettings(
            image_height=self.image_size,
            image_width=self.image_size,
            # image_height=self.h,
            # image_width=self.w,
            tanfovx=self.tanfov,
            tanfovy=self.tanfov,
            bg=self.bg * 0,
            scale_modifier=1.0,
            viewmatrix=w2c,
            projmatrix=full_proj,
            sh_degree=0,
            campos=cam_center,
            prefiltered=False,
            debug=False
        )
        self.rasterizer = GaussianRasterizer(raster_settings=self.raster_settings)
        self.rasterizer_bg = GaussianRasterizer(raster_settings=raster_settings_bg)

    def depths_to_points(self, depthmap):
        c2w = (self.w2c[0].T).inverse()
        W, H = self.image_size, self.image_size
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W) / 2],
            [0, H / 2, 0, (H) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ self.full_proj[0]
        intrins = (projection_matrix @ ndc2pix)[:3,:3].T
        
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
        rays_o = c2w[:3,3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
        return points

    def depth_to_normal(self, depth):
        """
            view: view camera
            depth: depthmap 
        """
        points = self.depths_to_points(depth).reshape(*depth.shape[1:], 3)
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output
        
    def render_gaussian(self, means3D, colors_precomp, rotations, opacities, scales, cov3D_precomp=None, label=None):
        '''
        mode: normal, phong, texture
        '''
        screenspace_points = (
            torch.zeros_like(
                means3D,
                dtype=means3D.dtype,
                requires_grad=True,
                device=means3D.device,
            )
            + 0
        )

        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        if cov3D_precomp != None:
            seg_color = (self.part_label.float() * 0.2).unsqueeze(-1).expand(-1, 3)
            # multi_layer
            # seg_color = torch.cat([seg_color, seg_color[-22152:]])
            # p_hom = torch.cat([means3D, torch.ones_like(means3D[...,:1])], -1).unsqueeze(-1)
            # p_view = torch.matmul(self.w2c[0].transpose(0,1), p_hom)
            # p_view = p_view[...,:3,:]
            # depth = p_view.squeeze()[...,2:3].repeat(1,3)
            # print(depth.min())
            # print(depth.max())
            # assert False
            # full_mask = label.sum(-1).bool()
            # full_mask = (full_mask * (opacities>0.005)[:, 0]).detach()
            # cloth_mask = label[:, 1].bool()
            # hair_mask = label[:, 2].bool()
            # label_all = label.clone()
            # label_all[body_mask] = 1 * 70 / 255
            # label_all[cloth_mask] = 2 * 70 / 255
            # label_all[hair_mask] = 3 * 70 / 255
            # opacity_mask = (opacities > 0.1)[:, 0]
            # hair_mask = hair_mask * opacity_mask
            # image_, _, _, _ = self.rasterizer(means3D=means3D[full_mask], colors_precomp=colors_precomp[full_mask], \
            #     opacities=opacities[full_mask], means2D=screenspace_points[full_mask], cov3D_precomp=cov3D_precomp[full_mask])
            # seg_, _, _, _ = self.rasterizer_bg(means3D=means3D[full_mask], colors_precomp=label_all[full_mask], \
            #     opacities=opacities[full_mask], means2D=screenspace_points[full_mask], cov3D_precomp=cov3D_precomp[full_mask])
            # image_body, _, _, alpha_body = self.rasterizer(means3D=means3D[body_mask], colors_precomp=colors_precomp[body_mask], \
            #     opacities=opacities[body_mask], means2D=screenspace_points[body_mask], cov3D_precomp=cov3D_precomp[body_mask])
            # image_cloth, _, _, alpha_cloth = self.rasterizer(means3D=means3D[cloth_mask], colors_precomp=colors_precomp[cloth_mask], \
            #     opacities=opacities[cloth_mask], means2D=screenspace_points[cloth_mask], cov3D_precomp=cov3D_precomp[cloth_mask])
            # image_hair, _, _, alpha_hair = self.rasterizer(means3D=means3D[hair_mask], colors_precomp=colors_precomp[hair_mask], \
            #     opacities=opacities[hair_mask], means2D=screenspace_points[hair_mask], cov3D_precomp=cov3D_precomp[hair_mask])
            body_mask, up_mask, hair_mask, shoes_mask, low_mask = self.part_mask.clone().chunk(5)
            # multi_layer mask
            # device = body_mask.device
            # body_mask = torch.cat([body_mask, torch.zeros(22152).to(device).bool()])
            # up_mask = torch.cat([up_mask, torch.ones(22152).to(device).bool()])
            # hair_mask = torch.cat([hair_mask, torch.zeros(22152).to(device).bool()])
            # shoes_mask = torch.cat([shoes_mask, torch.zeros(22152).to(device).bool()])
            # low_mask = torch.cat([low_mask, torch.zeros(22152).to(device).bool()])

            opacity_mask = (opacities.clone()>0.05)[..., 0].detach()
            body_mask *= opacity_mask
            up_mask *= opacity_mask
            hair_mask *= opacity_mask
            shoes_mask *= opacity_mask
            low_mask *= opacity_mask
            # component_mask = up_mask + shoes_mask
            # component_mask *= opacity_mask

            image_, _, _, alpha = self.rasterizer(means3D=means3D[opacity_mask], colors_precomp=colors_precomp[opacity_mask], \
                opacities=opacities[opacity_mask], means2D=screenspace_points[opacity_mask], cov3D_precomp=cov3D_precomp[opacity_mask])
            # image_, _, _, alpha = self.rasterizer(means3D=means3D[opacity_mask], colors_precomp=colors_precomp[opacity_mask], \
            #     opacities=opacities[opacity_mask], means2D=screenspace_points[opacity_mask], cov3D_precomp=cov3D_precomp[opacity_mask])
            # image_, _, _, alpha = self.rasterizer(means3D=means3D[component_mask], colors_precomp=colors_precomp[component_mask], \
            #     opacities=opacities[component_mask], means2D=screenspace_points[component_mask], cov3D_precomp=cov3D_precomp[component_mask])
            # depth_
            # multi_layer
            seg_, _, _, _ = self.rasterizer_bg(means3D=means3D[opacity_mask], colors_precomp=seg_color[opacity_mask], \
                opacities=opacities[opacity_mask], means2D=screenspace_points[opacity_mask], cov3D_precomp=cov3D_precomp[opacity_mask])
            image_body, _, _, alpha_body = self.rasterizer(means3D=means3D[body_mask], colors_precomp=colors_precomp[body_mask], \
                opacities=opacities[body_mask], means2D=screenspace_points[body_mask], cov3D_precomp=cov3D_precomp[body_mask])
            _, _, _, alpha_inner = self.rasterizer(means3D=means3D[body_mask], colors_precomp=colors_precomp[body_mask], \
                opacities=opacities[body_mask].detach()*0+1, means2D=screenspace_points[body_mask], cov3D_precomp=cov3D_precomp[body_mask])
            image_cloth, _, _, alpha_cloth = self.rasterizer(means3D=means3D[up_mask], colors_precomp=colors_precomp[up_mask], \
                opacities=opacities[up_mask], means2D=screenspace_points[up_mask], cov3D_precomp=cov3D_precomp[up_mask])
            # image_l1, _, _, alpha_l1 = self.rasterizer(means3D=torch.cat([means3D[hair_mask], means3D[shoes_mask]], dim=0), colors_precomp=torch.cat([colors_precomp[hair_mask],colors_precomp[shoes_mask]], dim=0), \
            #     opacities=torch.cat([opacities[hair_mask], opacities[shoes_mask]], dim=0), means2D=torch.cat([screenspace_points[hair_mask], screenspace_points[shoes_mask]], dim=0), cov3D_precomp=torch.cat([cov3D_precomp[hair_mask], cov3D_precomp[shoes_mask]], dim=0))
            image_hair, _, _, alpha_hair = self.rasterizer(means3D=means3D[hair_mask], colors_precomp=colors_precomp[hair_mask], \
                opacities=opacities[hair_mask], means2D=screenspace_points[hair_mask], cov3D_precomp=cov3D_precomp[hair_mask])
            image_shoes, _, _, alpha_shoes = self.rasterizer(means3D=means3D[shoes_mask], colors_precomp=colors_precomp[shoes_mask], \
                opacities=opacities[shoes_mask], means2D=screenspace_points[shoes_mask], cov3D_precomp=cov3D_precomp[shoes_mask])
            image_low, _, _, alpha_low = self.rasterizer(means3D=means3D[low_mask], colors_precomp=colors_precomp[low_mask], \
                opacities=opacities[low_mask], means2D=screenspace_points[low_mask], cov3D_precomp=cov3D_precomp[low_mask])
            # image_dress, _, _, alpha_dress = self.rasterizer(means3D=torch.cat([means3D[up_mask], means3D[low_mask]], dim=0), colors_precomp=torch.cat([colors_precomp[up_mask],colors_precomp[low_mask]], dim=0), \
            #     opacities=torch.cat([opacities[up_mask], opacities[low_mask]], dim=0), means2D=torch.cat([screenspace_points[up_mask], screenspace_points[low_mask]], dim=0), cov3D_precomp=torch.cat([cov3D_precomp[up_mask], cov3D_precomp[low_mask]], dim=0))
            # normal_body = (self.depth_to_normal(image_body[:1]) * 0.5) + 0.5 # [-1, 1]
            # normal_body = normal_body * alpha_body.permute(1, 2, 0) + self.bg.reshape(1, 1, 3) * (1-alpha_body.permute(1, 2, 0))
            # depth_all = torch.cat([depth_body, depth_cloth, depth_hair, depth_shoes, depth_low])
            # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/image_body.png', (image_body.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
            # cv2.imwrite('/home/zhangweitian/HighResAvatar/debug/seg_.png', (seg_.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8))
            # cv2.imwrite('/mnt/sdb/zwt/LayerAvatar/debug/normal_body.png', ((normal_body).detach().cpu().numpy()*255).astype(np.uint8))
            # assert False
            # seg = torch.cat([alpha, seg_, alpha_body, alpha_cloth, alpha_hair, alpha_shoes, alpha_low], dim=0)
            seg = torch.cat([alpha_inner, seg_, alpha_body, alpha_cloth, alpha_hair, alpha_shoes, alpha_low], dim=0)
            image = torch.cat([image_, image_body, image_cloth, image_hair, image_shoes, image_low], dim=0)
            # image = image_
            # seg = alpha
            # image = image_cloth
            # seg = alpha_cloth
            # image = torch.cat([image_, image_body, image_cloth, image_low, image_l1], dim=0)
            # seg = torch.cat([alpha, seg_, alpha_body, alpha_cloth, alpha_low, alpha_l1], dim=0)
            # depth = torch.cat([depth_, depth_body[:1], depth_cloth[:1], depth_hair[:1], depth_shoes[:1], depth_low[:1]], dim=0)
        else:
            assert False
            # print(is_rotation_matrix(rotations[0]))
            # assert False
            # rotations = quat_xyzw_to_wxyz(rotmat_to_unitquat(rotations)) 
            # quaternion = matrix_to_quaternion(rotations)
            # mat = quaternion_to_matrix(quaternion)
            # print(rotations.shape)
            # print(mat.shape)
            # print(rotations[:2])
            # print(mat[:2])
            # assert False
            # print(rotations.shape)
            # assert False
            full_mask = label.sum(-1).bool()
            # full_mask = (full_mask * (opacities>0.005)[:, 0]).detach()
            body_mask = label[:, 0].bool()
            cloth_mask = label[:, 1].bool()
            hair_mask = label[:, 2].bool()
            label_all = label.clone()
            label_all[body_mask] = 1 * 70 / 255
            label_all[cloth_mask] = 2 * 70 / 255
            label_all[hair_mask] = 3 * 70 / 255
            # opacity_mask = (opacities > 0.1)[:, 0]
            # hair_mask = hair_mask * opacity_mask
            image_, _, _, _ = self.rasterizer(means3D=means3D[full_mask], colors_precomp=colors_precomp[full_mask], \
                opacities=opacities[full_mask], means2D=screenspace_points[full_mask], scales=scales[full_mask], rotations=rotations[full_mask])
            seg_, _, _, _ = self.rasterizer_bg(means3D=means3D[full_mask], colors_precomp=label_all[full_mask], \
                opacities=opacities[full_mask], means2D=screenspace_points[full_mask], scales=scales[full_mask], rotations=rotations[full_mask])
            image_body, _, _, alpha_body = self.rasterizer(means3D=means3D[body_mask], colors_precomp=colors_precomp[body_mask], \
                opacities=opacities[body_mask], means2D=screenspace_points[body_mask], scales=scales[body_mask], rotations=rotations[body_mask])
            image_cloth, _, _, alpha_cloth = self.rasterizer(means3D=means3D[cloth_mask], colors_precomp=colors_precomp[cloth_mask], \
                opacities=opacities[cloth_mask], means2D=screenspace_points[cloth_mask], scales=scales[cloth_mask], rotations=rotations[cloth_mask])
            image_hair, _, _, alpha_hair = self.rasterizer(means3D=means3D[hair_mask], colors_precomp=colors_precomp[hair_mask], \
                opacities=opacities[hair_mask], means2D=screenspace_points[hair_mask], scales=scales[hair_mask], rotations=rotations[hair_mask])
            seg = torch.cat([seg_, alpha_cloth, alpha_body, alpha_hair], dim=0)
            image = torch.cat([image_, image_cloth, image_body, image_hair], dim=0)
            # depth_index = torch.argsort(torch.stack([depth_hair, depth_cloth, depth_body]), dim=0)
            # composite_image = torch.stack([image_hair, image_cloth, image_body])
            # composite_alpha = torch.stack([alpha_hair, alpha_cloth, alpha_body])
            # composite_seg = torch.stack([alpha_hair*self.label_value[2], alpha_cloth*self.label_value[1], alpha_body*self.label_value[0]])
            # composite_all = torch.gather(torch.cat([composite_image, composite_seg, composite_alpha], dim=1), 0, depth_index.expand(-1, 5, -1, -1))
            # composite_attr, compo_alpha = composite_all.split([4, 1], dim=1)
            # alphas_shifted = torch.cat([torch.ones_like(compo_alpha[:1]), 1-compo_alpha], 0)
            # weights = compo_alpha * torch.cumprod(alphas_shifted, dim=0)[:-1]
            # image_seg = torch.sum(composite_attr * weights, dim=0)
            # seg = torch.cat([image_seg[3:], alpha_cloth, alpha_body, alpha_hair], dim=0)
            # image = torch.cat([image_seg[:3], image_cloth, image_body, image_hair], dim=0)
            # image, _ = self.rasterizer(means3D=means3D, colors_precomp=colors_precomp, \
            #     rotations=torch.nn.functional.normalize(rotations), opacities=opacities, scales=scales, \
            #     means2D=screenspace_points)
        
        # return  torch.cat([image, seg], dim=0)
        return  image, seg