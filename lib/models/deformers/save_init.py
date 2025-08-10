import trimesh
import torch
from smplx import SMPL
import numpy as np

# def init_point(w=128, h=128, d=128//4):
#     x_range = (torch.linspace(-1,1,steps=w)).view(1, 1, 1, w).expand(1, d, h, w)  # [1, H, W, D]
#     y_range = (torch.linspace(-1,1,steps=h)).view(1, 1, h, 1).expand(1, d, h, w)  # [1, H, W, D]
#     z_range = (torch.linspace(-1,1,steps=d)).view(1, d, 1, 1).expand(1, d, h, w)  # [1, H, W, D]
#     grid = torch.cat((x_range, y_range, z_range), dim=0).reshape(1, 3, -1).permute(0,2,1)
#     grid[:,:,-1] /= 4
#     return grid


body_model = SMPL('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/smplx/SMPLX', gender='neutral', \
                                create_body_pose=False, \
                                create_betas=False, \
                                create_global_orient=False, \
                                create_transl=False)

body_pose_t = torch.zeros((1, 69))
transl = torch.zeros((1, 3))
transl[:,1] = 0.3
global_orient = torch.zeros((1, 3))
betas = torch.zeros((1, 10))
smpl_outputs = body_model(betas=betas, body_pose=body_pose_t, transl=transl, global_orient = global_orient)

verts = smpl_outputs.vertices
smpl_faces = torch.as_tensor(body_model.faces.astype(np.int64))
new_verts, new_faces = trimesh.remesh.subdivide(verts[0], smpl_faces)
new_verts, new_faces = trimesh.remesh.subdivide(new_verts, new_faces)
print(new_verts.shape)
print(new_faces.shape)
new_verts = torch.as_tensor(new_verts)
new_verts += 0.03 * torch.randn(new_verts.shape)
round_pcd = torch.as_tensor(np.load('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_new.npy'))
select_idx = np.random.choice(round_pcd.shape[0], 10000, replace=False)
select_pcd = round_pcd[select_idx]
final_init_pcd = torch.cat([new_verts, select_pcd], dim=0)
np.save('/mnt/sdb/zwt/SSDNeRF/work_dirs/cache/init_pcd_smpl.npy', final_init_pcd.numpy())
new_mesh = trimesh.Trimesh(new_verts, new_faces)
new_mesh.export('/mnt/sdb/zwt/SSDNeRF/lib/models/deformers/dense_smpl.obj')



