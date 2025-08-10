from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
import torch
import torch.nn as nn
import math
from pytorch3d.transforms import matrix_to_quaternion
# from pytorch3d.transforms import quaternion_to_matrix


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

# def getProjectionMatrix(znear, zfar, fovX, fovY, K = None, img_h = None, img_w = None):
#     if K is None:
#         tanHalfFovY = math.tan((fovY / 2))
#         tanHalfFovX = math.tan((fovX / 2))
#         top = tanHalfFovY * znear
#         bottom = -top
#         right = tanHalfFovX * znear
#         left = -right
#     else:
#         near_fx = znear / K[0, 0]
#         near_fy = znear / K[1, 1]

#         left = - (img_w - K[0, 2]) * near_fx
#         right = K[0, 2] * near_fx
#         bottom = (K[1, 2] - img_h) * near_fy
#         top = K[1, 2] * near_fy

#     P = torch.zeros(4, 4)

#     z_sign = 1.0

    # P[0, 0] = 2.0 * znear / (right - left)
    # P[1, 1] = 2.0 * znear / (top - bottom)
    # P[0, 2] = (right + left) / (right - left)
    # P[1, 2] = (top + bottom) / (top - bottom)
    # P[3, 2] = z_sign
    # P[2, 2] = z_sign * zfar / (zfar - znear)
    # P[2, 3] = -(zfar * znear) / (zfar - znear)
    # return P


class GRenderer(nn.Module):
    def __init__(self, image_size=256, anti_alias=False, f=5000, near=0.01, far=40, bg_color=0):
        super().__init__()

        self.anti_alias = anti_alias
        self.image_size = image_size
        # self.h = 1024
        # self.w = 750
        # fx = 5000
        # fy = 5000
        # 0001
        # fx = 1254.1212158203125 / 2
        # fy = 1254.441162109375 / 2
        # fx = 1254.603515625 / 2
        # fy = 1254.95751953125 / 2
        # 0000
        # fx = 1253.78564453125 / 2
        # fy = 1254.04443359375 / 2
       
        # FoVx = focal2fov(f, self.image_size)
        # FoVy = focal2fov(f, self.image_size)
        # self.tanfovx = math.tan(FoVx * 0.5)
        # self.tanfovy = math.tan(FoVy * 0.5)
        self.tanfov = self.image_size / (2 * f)

        if bg_color == 0:
            bg = torch.tensor([0, 0, 0], dtype=torch.float32)
        else:
            bg = torch.tensor([1, 1, 1], dtype=torch.float32)

        self.register_buffer('bg', bg)
        
        # opengl_proj = getProjectionMatrix(znear = 0.1, zfar = 100, fovX = FoVx, fovY = FoVy).transpose(0, 1).cuda()
        opengl_proj = torch.tensor([[2 * f / self.image_size, 0.0, 0.0, 0.0],
                                    [0.0, 2 * f / self.image_size, 0.0, 0.0],
                                    [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                    [0.0, 0.0, 1.0, 0.0]]).float().unsqueeze(0).transpose(1, 2)
        self.register_buffer('opengl_proj', opengl_proj, persistent=False)

        if anti_alias: image_size = image_size*2
        
    # def prepare(self, cameras, i):
    def prepare(self, cameras):
        # cam_center = cameras[0][i]
        # w2c = cameras[1][i]
        # right for train

        # cam_center = cameras[:3]
        # fx, fy = cameras[3], cameras[4]
        # fx_ = fx / 2
        # fy_ = fy / 2
        # w2c = cameras[5:].reshape(4, 4)
        # FoVx = focal2fov(fx_, self.w)
        # FoVy = focal2fov(fy_, self.h)
        # tanfovx = math.tan(FoVx * 0.5)
        # tanfovy = math.tan(FoVy * 0.5)
        # opengl_proj = getProjectionMatrix(znear = 0.1, zfar = 100, fovX = FoVx, fovY = FoVy).transpose(0, 1).cuda()
        # w2c = w2c.transpose(0, 1)
        cam_center = cameras[:3]
        w2c = cameras[3:].reshape(4, 4)
        w2c = w2c.unsqueeze(0).transpose(1, 2)
        full_proj = w2c.bmm(self.opengl_proj)
        # full_proj = w2c.unsqueeze(0).bmm(opengl_proj.unsqueeze(0))
        self.full_proj = full_proj
        # cam = Camera(
        # bg_white = torch.ones_like(self.bg)
        self.raster_settings = GaussianRasterizationSettings(
            image_height=self.image_size,
            image_width=self.image_size,
            # image_height=self.h,
            # image_width=self.w,
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
        
    def render_gaussian(self, means3D, colors_precomp, rotations, opacities, scales, cov3D_precomp=None, label=None):
        '''
        mode: normal, phong, texture
        '''
        assert label is not None
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
        
        # R = quaternion_to_matrix(torch.nn.functional.normalize(rotations))
        # R_def = tfs[:,:3,:3] @ R
        # cov3D_precomp = get_covariance(scales, R_def)
        # image, _ = self.rasterizer(means3D=means3D, colors_precomp=colors_precomp, \
        #     rotations=torch.nn.functional.normalize(rotations), opacities=opacities, scales=scales, \
        #     means2D=screenspace_points)
        if cov3D_precomp != None:
            # opacities_ = opacities.clone()
            # opacities_[:50200] *= 0
            # opacities_[:50200] += 1
            # body_mask = torch.zeros_like(opacities)
            # cloth_mask = torch.zeros_like(opacities)
            # hair_mask = torch.zeros_like(opacities)
            full_mask = label.sum(-1).bool()
            # full_mask = (full_mask * (opacities>0.005)[:, 0]).detach()
            body_mask = label[:, 0].bool()
            cloth_mask = label[:, 1].bool()
            hair_mask = label[:, 2].bool()
            # body_mask[:50200] = 1
            # body_mask = body_mask.bool()[:, 0]
            # cloth_mask[50200:50200*2] = 1
            # cloth_mask = cloth_mask.bool()[:, 0]
            # hair_mask[-50200:] = 1
            # hair_mask = hair_mask.bool()[:, 0]
            label_all = label.clone()
            label_all[body_mask] = 1 * 70 / 255
            label_all[cloth_mask] = 2 * 70 / 255
            label_all[hair_mask] = 3 * 70 / 255
            # opacity_mask = (opacities > 0.1)[:, 0]
            # hair_mask = hair_mask * opacity_mask
            image_, _ = self.rasterizer(means3D=means3D[full_mask], colors_precomp=colors_precomp[full_mask], \
                opacities=opacities[full_mask], means2D=screenspace_points[full_mask], cov3D_precomp=cov3D_precomp[full_mask])
            image_body, _ = self.rasterizer(means3D=means3D[body_mask], colors_precomp=colors_precomp[body_mask], \
                opacities=opacities[body_mask], means2D=screenspace_points[body_mask], cov3D_precomp=cov3D_precomp[body_mask])
            image_cloth, _ = self.rasterizer(means3D=means3D[cloth_mask], colors_precomp=colors_precomp[cloth_mask], \
                opacities=opacities[cloth_mask], means2D=screenspace_points[cloth_mask], cov3D_precomp=cov3D_precomp[cloth_mask])
            image_hair, _ = self.rasterizer(means3D=means3D[hair_mask], colors_precomp=colors_precomp[hair_mask], \
                opacities=opacities[hair_mask], means2D=screenspace_points[hair_mask], cov3D_precomp=cov3D_precomp[hair_mask])
            seg_, _ = self.rasterizer_bg(means3D=means3D[full_mask], colors_precomp=label_all[full_mask], \
                opacities=opacities[full_mask], means2D=screenspace_points[full_mask], cov3D_precomp=cov3D_precomp[full_mask])
            seg_body, _ = self.rasterizer_bg(means3D=means3D[body_mask], colors_precomp=label[body_mask]*0+1, \
                opacities=opacities[body_mask], means2D=screenspace_points[body_mask], cov3D_precomp=cov3D_precomp[body_mask])
            # label_cloth = label.clone()
            # label_cloth[:50200] *= 0
            # label_cloth[-50200:] *= 0
            seg_cloth, _ = self.rasterizer_bg(means3D=means3D[cloth_mask], colors_precomp=label[cloth_mask]*0+1, \
                opacities=opacities[cloth_mask], means2D=screenspace_points[cloth_mask], cov3D_precomp=cov3D_precomp[cloth_mask])
            # seg_cloth, _ = self.rasterizer_bg(means3D=means3D[~body_mask], colors_precomp=label_cloth[~body_mask], \
            #     opacities=opacities_[~body_mask], means2D=screenspace_points[~body_mask], cov3D_precomp=cov3D_precomp[~body_mask])
            # opacities_hair = opacities.clone()
            # opacities_hair[:-50200] *= 0
            # label_hair = label.clone()
            # label_hair[:-50200] *= 0
            seg_hair, _ = self.rasterizer_bg(means3D=means3D[hair_mask], colors_precomp=label[hair_mask]*0+1, \
                opacities=opacities[hair_mask], means2D=screenspace_points[hair_mask], cov3D_precomp=cov3D_precomp[hair_mask])
            # seg_hair, _ = self.rasterizer_bg(means3D=means3D[~body_mask], colors_precomp=label_hair[~body_mask], \
            #     opacities=opacities_[~body_mask], means2D=screenspace_points[~body_mask], cov3D_precomp=cov3D_precomp[~body_mask])
            seg = torch.cat([seg_, seg_cloth, seg_body, seg_hair], dim=0)
            image = torch.cat([image_, image_cloth, image_body, image_hair], dim=0)
        else:
            rotations = matrix_to_quaternion(rotations)
            full_mask = label.sum(-1).bool()
            body_mask = label[:, 0].bool()
            cloth_mask = label[:, 1].bool()
            hair_mask = label[:, 2].bool()
            label_all = label.clone()
            label_all[body_mask] = 1 * 70 / 255
            label_all[cloth_mask] = 2 * 70 / 255
            label_all[hair_mask] = 3 * 70 / 255
           
            image_, _ = self.rasterizer(means3D=means3D[full_mask], colors_precomp=colors_precomp[full_mask], \
                opacities=opacities[full_mask], means2D=screenspace_points[full_mask], rotations=rotations[full_mask], scales=scales[full_mask])
            image_body, _ = self.rasterizer(means3D=means3D[body_mask], colors_precomp=colors_precomp[body_mask], \
                opacities=opacities[body_mask], means2D=screenspace_points[body_mask], rotations=rotations[body_mask], scales=scales[body_mask])
            image_cloth, _ = self.rasterizer(means3D=means3D[cloth_mask], colors_precomp=colors_precomp[cloth_mask], \
                opacities=opacities[cloth_mask], means2D=screenspace_points[cloth_mask], rotations=rotations[cloth_mask], scales=scales[cloth_mask])
            image_hair, _ = self.rasterizer(means3D=means3D[hair_mask], colors_precomp=colors_precomp[hair_mask], \
                opacities=opacities[hair_mask], means2D=screenspace_points[hair_mask], rotations=rotations[hair_mask], scales=scales[hair_mask])
            seg_, _ = self.rasterizer_bg(means3D=means3D[full_mask], colors_precomp=label_all[full_mask], \
                opacities=opacities[full_mask], means2D=screenspace_points[full_mask], rotations=rotations[full_mask], scales=scales[full_mask])
            seg_body, _ = self.rasterizer_bg(means3D=means3D[body_mask], colors_precomp=label[body_mask]*0+1, \
                opacities=opacities[body_mask], means2D=screenspace_points[body_mask], rotations=rotations[body_mask], scales=scales[body_mask])
            seg_cloth, _ = self.rasterizer_bg(means3D=means3D[cloth_mask], colors_precomp=label[cloth_mask]*0+1, \
                opacities=opacities[cloth_mask], means2D=screenspace_points[cloth_mask], rotations=rotations[cloth_mask], scales=scales[cloth_mask])
            seg_hair, _ = self.rasterizer_bg(means3D=means3D[hair_mask], colors_precomp=label[hair_mask]*0+1, \
                opacities=opacities[hair_mask], means2D=screenspace_points[hair_mask], rotations=rotations[hair_mask], scales=scales[hair_mask])
            seg = torch.cat([seg_, seg_cloth, seg_body, seg_hair], dim=0)
            image = torch.cat([image_, image_cloth, image_body, image_hair], dim=0)
        assert False
        return  image, seg